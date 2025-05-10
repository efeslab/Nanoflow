// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ib.hpp"

#include <infiniband/verbs.h>
#include <malloc.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <mscclpp/core.hpp>
#include <mscclpp/fifo.hpp>
#include <sstream>
#include <string>

#include "api.h"
#include "debug.h"

#if !defined(__HIP_PLATFORM_AMD__)

// Check if nvidia_peermem kernel module is loaded
static bool checkNvPeerMemLoaded() {
  std::ifstream file("/proc/modules");
  std::string line;
  while (std::getline(file, line)) {
    if (line.find("nvidia_peermem") != std::string::npos) return true;
  }
  return false;
}

#endif  // !defined(__HIP_PLATFORM_AMD__)

namespace mscclpp {

IbMr::IbMr(ibv_pd* pd, void* buff, std::size_t size) : buff(buff) {
  if (size == 0) {
    throw std::invalid_argument("invalid size: " + std::to_string(size));
  }
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) {
    pageSize = sysconf(_SC_PAGESIZE);
  }
  uintptr_t addr = reinterpret_cast<uintptr_t>(buff) & -pageSize;
  std::size_t pages = (size + (reinterpret_cast<uintptr_t>(buff) - addr) + pageSize - 1) / pageSize;
  this->mr = ibv_reg_mr(pd, reinterpret_cast<void*>(addr), pages * pageSize,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                            IBV_ACCESS_RELAXED_ORDERING | IBV_ACCESS_REMOTE_ATOMIC);
  if (this->mr == nullptr) {
    std::stringstream err;
    err << "ibv_reg_mr failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  this->size = pages * pageSize;
}

IbMr::~IbMr() { ibv_dereg_mr(this->mr); }

IbMrInfo IbMr::getInfo() const {
  IbMrInfo info;
  info.addr = reinterpret_cast<uint64_t>(this->buff);
  info.rkey = this->mr->rkey;
  return info;
}

const void* IbMr::getBuff() const { return this->buff; }

uint32_t IbMr::getLkey() const { return this->mr->lkey; }

IbQp::IbQp(ibv_context* ctx, ibv_pd* pd, int port, int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr,
           int maxWrPerSend)
    : numSignaledPostedItems(0), numSignaledStagedItems(0), maxCqPollNum(maxCqPollNum), maxWrPerSend(maxWrPerSend) {
  this->cq = ibv_create_cq(ctx, maxCqSize, nullptr, nullptr, 0);
  if (this->cq == nullptr) {
    std::stringstream err;
    err << "ibv_create_cq failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }

  struct ibv_qp_init_attr qpInitAttr;
  std::memset(&qpInitAttr, 0, sizeof(qpInitAttr));
  qpInitAttr.sq_sig_all = 0;
  qpInitAttr.send_cq = this->cq;
  qpInitAttr.recv_cq = this->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.cap.max_send_wr = maxSendWr;
  qpInitAttr.cap.max_recv_wr = maxRecvWr;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = 0;

  struct ibv_qp* _qp = ibv_create_qp(pd, &qpInitAttr);
  if (_qp == nullptr) {
    std::stringstream err;
    err << "ibv_create_qp failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }

  struct ibv_port_attr portAttr;
  if (ibv_query_port(ctx, port, &portAttr) != 0) {
    std::stringstream err;
    err << "ibv_query_port failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  this->info.lid = portAttr.lid;
  this->info.port = port;
  this->info.linkLayer = portAttr.link_layer;
  this->info.qpn = _qp->qp_num;
  this->info.mtu = portAttr.active_mtu;
  this->info.is_grh = (portAttr.flags & IBV_QPF_GRH_REQUIRED);

  if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND || this->info.is_grh) {
    union ibv_gid gid;
    if (ibv_query_gid(ctx, port, 0, &gid) != 0) {
      std::stringstream err;
      err << "ibv_query_gid failed (errno " << errno << ")";
      throw mscclpp::IbError(err.str(), errno);
    }
    this->info.spn = gid.global.subnet_prefix;
    this->info.iid = gid.global.interface_id;
  }

  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = port;
  qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  if (ibv_modify_qp(_qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
    std::stringstream err;
    err << "ibv_modify_qp failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  this->qp = _qp;
  this->wrn = 0;
  this->wrs = std::make_unique<ibv_send_wr[]>(maxWrPerSend);
  this->sges = std::make_unique<ibv_sge[]>(maxWrPerSend);
  this->wcs = std::make_unique<ibv_wc[]>(maxCqPollNum);
}

IbQp::~IbQp() {
  ibv_destroy_qp(this->qp);
  ibv_destroy_cq(this->cq);
}

void IbQp::rtr(const IbQpInfo& info) {
  struct ibv_qp_attr qp_attr;
  std::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.path_mtu = static_cast<ibv_mtu>(info.mtu);
  qp_attr.dest_qp_num = info.qpn;
  qp_attr.rq_psn = 0;
  qp_attr.max_dest_rd_atomic = 1;
  qp_attr.min_rnr_timer = 0x12;
  if (info.linkLayer == IBV_LINK_LAYER_ETHERNET || info.is_grh) {
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.dgid.global.subnet_prefix = info.spn;
    qp_attr.ah_attr.grh.dgid.global.interface_id = info.iid;
    qp_attr.ah_attr.grh.flow_label = 0;
    qp_attr.ah_attr.grh.sgid_index = 0;
    qp_attr.ah_attr.grh.hop_limit = 255;
    qp_attr.ah_attr.grh.traffic_class = 0;
  } else {
    qp_attr.ah_attr.is_global = 0;
  }
  qp_attr.ah_attr.dlid = info.lid;
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = info.port;
  int ret = ibv_modify_qp(this->qp, &qp_attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (ret != 0) {
    std::stringstream err;
    err << "ibv_modify_qp failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
}

void IbQp::rts() {
  struct ibv_qp_attr qp_attr;
  std::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.timeout = 18;
  qp_attr.retry_cnt = 7;
  qp_attr.rnr_retry = 7;
  qp_attr.sq_psn = 0;
  qp_attr.max_rd_atomic = 1;
  int ret = ibv_modify_qp(
      this->qp, &qp_attr,
      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  if (ret != 0) {
    std::stringstream err;
    err << "ibv_modify_qp failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
}

IbQp::WrInfo IbQp::getNewWrInfo() {
  if (this->wrn >= this->maxWrPerSend) {
    std::stringstream err;
    err << "too many outstanding work requests. limit is " << this->maxWrPerSend;
    throw mscclpp::Error(err.str(), ErrorCode::InvalidUsage);
  }
  int wrn = this->wrn;

  ibv_send_wr* wr_ = &this->wrs[wrn];
  ibv_sge* sge_ = &this->sges[wrn];
  wr_->sg_list = sge_;
  wr_->num_sge = 1;
  wr_->next = nullptr;
  if (wrn > 0) {
    this->wrs[wrn - 1].next = wr_;
  }
  this->wrn++;
  return IbQp::WrInfo{wr_, sge_};
}

void IbQp::stageSend(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                     uint64_t dstOffset, bool signaled) {
  auto wrInfo = this->getNewWrInfo();
  wrInfo.wr->wr_id = wrId;
  wrInfo.wr->opcode = IBV_WR_RDMA_WRITE;
  wrInfo.wr->send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wrInfo.wr->wr.rdma.remote_addr = (uint64_t)(info.addr) + dstOffset;
  wrInfo.wr->wr.rdma.rkey = info.rkey;
  wrInfo.sge->addr = (uint64_t)(mr->getBuff()) + srcOffset;
  wrInfo.sge->length = size;
  wrInfo.sge->lkey = mr->getLkey();
  if (signaled) (this->numSignaledStagedItems)++;
}

void IbQp::stageAtomicAdd(const IbMr* mr, const IbMrInfo& info, uint64_t wrId, uint64_t dstOffset, uint64_t addVal,
                          bool signaled) {
  auto wrInfo = this->getNewWrInfo();
  wrInfo.wr->wr_id = wrId;
  wrInfo.wr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wrInfo.wr->send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wrInfo.wr->wr.atomic.remote_addr = (uint64_t)(info.addr) + dstOffset;
  wrInfo.wr->wr.atomic.rkey = info.rkey;
  wrInfo.wr->wr.atomic.compare_add = addVal;
  wrInfo.sge->addr = (uint64_t)(mr->getBuff());
  wrInfo.sge->length = sizeof(uint64_t);  // atomic op is always on uint64_t
  wrInfo.sge->lkey = mr->getLkey();
  if (signaled) (this->numSignaledStagedItems)++;
}

void IbQp::stageSendWithImm(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                            uint64_t dstOffset, bool signaled, unsigned int immData) {
  auto wrInfo = this->getNewWrInfo();
  wrInfo.wr->wr_id = wrId;
  wrInfo.wr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wrInfo.wr->send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wrInfo.wr->wr.rdma.remote_addr = (uint64_t)(info.addr) + dstOffset;
  wrInfo.wr->wr.rdma.rkey = info.rkey;
  wrInfo.wr->imm_data = immData;
  wrInfo.sge->addr = (uint64_t)(mr->getBuff()) + srcOffset;
  wrInfo.sge->length = size;
  wrInfo.sge->lkey = mr->getLkey();
  if (signaled) (this->numSignaledStagedItems)++;
}

void IbQp::postSend() {
  if (this->wrn == 0) {
    return;
  }
  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(this->qp, this->wrs.get(), &bad_wr);
  if (ret != 0) {
    std::stringstream err;
    err << "ibv_post_send failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  this->wrn = 0;
  this->numSignaledPostedItems += this->numSignaledStagedItems;
  this->numSignaledStagedItems = 0;
  if (this->numSignaledPostedItems + 4 > this->cq->cqe) {
    WARN("IB: CQ is almost full ( %d / %d ). The connection needs to be flushed to prevent timeout errors.",
         this->numSignaledPostedItems, this->cq->cqe);
  }
}

int IbQp::pollCq() {
  int wcNum = ibv_poll_cq(this->cq, this->maxCqPollNum, this->wcs.get());
  if (wcNum > 0) {
    this->numSignaledPostedItems -= wcNum;
  }
  return wcNum;
}

IbQpInfo& IbQp::getInfo() { return this->info; }

const ibv_wc* IbQp::getWc(int idx) const { return &this->wcs[idx]; }

int IbQp::getNumCqItems() const { return this->numSignaledPostedItems; }

IbCtx::IbCtx(const std::string& devName) : devName(devName) {
#if !defined(__HIP_PLATFORM_AMD__)
  if (!checkNvPeerMemLoaded()) {
    throw mscclpp::Error("nvidia_peermem kernel module is not loaded", ErrorCode::InternalError);
  }
#endif  // !defined(__HIP_PLATFORM_AMD__)
  int num;
  struct ibv_device** devices = ibv_get_device_list(&num);
  for (int i = 0; i < num; ++i) {
    if (std::string(devices[i]->name) == devName) {
      this->ctx = ibv_open_device(devices[i]);
      break;
    }
  }
  ibv_free_device_list(devices);
  if (this->ctx == nullptr) {
    std::stringstream err;
    err << "ibv_open_device failed (errno " << errno << ", device name << " << devName << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  this->pd = ibv_alloc_pd(this->ctx);
  if (this->pd == nullptr) {
    std::stringstream err;
    err << "ibv_alloc_pd failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
}

IbCtx::~IbCtx() {
  this->mrs.clear();
  this->qps.clear();
  if (this->pd != nullptr) {
    ibv_dealloc_pd(this->pd);
  }
  if (this->ctx != nullptr) {
    ibv_close_device(this->ctx);
  }
}

bool IbCtx::isPortUsable(int port) const {
  struct ibv_port_attr portAttr;
  if (ibv_query_port(this->ctx, port, &portAttr) != 0) {
    std::stringstream err;
    err << "ibv_query_port failed (errno " << errno << ", port << " << port << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  return portAttr.state == IBV_PORT_ACTIVE &&
         (portAttr.link_layer == IBV_LINK_LAYER_ETHERNET || portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND);
}

int IbCtx::getAnyActivePort() const {
  struct ibv_device_attr devAttr;
  if (ibv_query_device(this->ctx, &devAttr) != 0) {
    std::stringstream err;
    err << "ibv_query_device failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  for (uint8_t port = 1; port <= devAttr.phys_port_cnt; ++port) {
    if (this->isPortUsable(port)) {
      return port;
    }
  }
  return -1;
}

IbQp* IbCtx::createQp(int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr, int maxWrPerSend,
                      int port /*=-1*/) {
  if (port == -1) {
    port = this->getAnyActivePort();
    if (port == -1) {
      throw mscclpp::Error("No active port found", ErrorCode::InternalError);
    }
  } else if (!this->isPortUsable(port)) {
    throw mscclpp::Error("invalid IB port: " + std::to_string(port), ErrorCode::InternalError);
  }
  qps.emplace_back(new IbQp(this->ctx, this->pd, port, maxCqSize, maxCqPollNum, maxSendWr, maxRecvWr, maxWrPerSend));
  return qps.back().get();
}

const IbMr* IbCtx::registerMr(void* buff, std::size_t size) {
  mrs.emplace_back(new IbMr(this->pd, buff, size));
  return mrs.back().get();
}

const std::string& IbCtx::getDevName() const { return this->devName; }

MSCCLPP_API_CPP int getIBDeviceCount() {
  int num;
  ibv_get_device_list(&num);
  return num;
}

std::string getHcaDevices(int deviceIndex) {
  const char* envValue = std::getenv("MSCCLPP_HCA_DEVICES");
  if (envValue) {
    std::vector<std::string> devices;
    std::string envStr(envValue);
    std::stringstream ss(envStr);
    std::string device;
    while (std::getline(ss, device, ',')) {
      devices.push_back(device);
    }
    if (deviceIndex >= (int)devices.size()) {
      throw std::invalid_argument("Not enough HCA devices are defined with MSCCLPP_HCA_DEVICES: " +
                                  std::string(envValue));
    }
    return devices[deviceIndex];
  }
  return "";
}

MSCCLPP_API_CPP std::string getIBDeviceName(Transport ibTransport) {
  int ibTransportIndex;
  switch (ibTransport) {  // TODO: get rid of this ugly switch
    case Transport::IB0:
      ibTransportIndex = 0;
      break;
    case Transport::IB1:
      ibTransportIndex = 1;
      break;
    case Transport::IB2:
      ibTransportIndex = 2;
      break;
    case Transport::IB3:
      ibTransportIndex = 3;
      break;
    case Transport::IB4:
      ibTransportIndex = 4;
      break;
    case Transport::IB5:
      ibTransportIndex = 5;
      break;
    case Transport::IB6:
      ibTransportIndex = 6;
      break;
    case Transport::IB7:
      ibTransportIndex = 7;
      break;
    default:
      throw std::invalid_argument("Not an IB transport");
  }
  std::string userHcaDevice = getHcaDevices(ibTransportIndex);
  if (!userHcaDevice.empty()) {
    return userHcaDevice;
  }

  int num;
  struct ibv_device** devices = ibv_get_device_list(&num);
  if (ibTransportIndex >= num) {
    std::stringstream ss;
    ss << "IB transport out of range: " << ibTransportIndex << " >= " << num;
    throw std::out_of_range(ss.str());
  }
  return devices[ibTransportIndex]->name;
}

MSCCLPP_API_CPP Transport getIBTransportByDeviceName(const std::string& ibDeviceName) {
  int num;
  struct ibv_device** devices = ibv_get_device_list(&num);
  for (int i = 0; i < num; ++i) {
    if (ibDeviceName == devices[i]->name) {
      switch (i) {  // TODO: get rid of this ugly switch
        case 0:
          return Transport::IB0;
        case 1:
          return Transport::IB1;
        case 2:
          return Transport::IB2;
        case 3:
          return Transport::IB3;
        case 4:
          return Transport::IB4;
        case 5:
          return Transport::IB5;
        case 6:
          return Transport::IB6;
        case 7:
          return Transport::IB7;
        default:
          throw std::out_of_range("IB device index out of range");
      }
    }
  }
  throw std::invalid_argument("IB device not found");
}

}  // namespace mscclpp

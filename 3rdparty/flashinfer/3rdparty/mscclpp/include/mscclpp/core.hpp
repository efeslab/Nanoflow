// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CORE_HPP_
#define MSCCLPP_CORE_HPP_

#define MSCCLPP_MAJOR 0
#define MSCCLPP_MINOR 5
#define MSCCLPP_PATCH 1
#define MSCCLPP_VERSION (MSCCLPP_MAJOR * 10000 + MSCCLPP_MINOR * 100 + MSCCLPP_PATCH)

#include <array>
#include <bitset>
#include <future>
#include <memory>
#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <string>
#include <vector>

#include "errors.hpp"

namespace mscclpp {

#define MSCCLPP_UNIQUE_ID_BYTES 128

/// Unique ID for a process. This is a MSCCLPP_UNIQUE_ID_BYTES byte array that uniquely identifies a process.
using UniqueId = std::array<uint8_t, MSCCLPP_UNIQUE_ID_BYTES>;

/// Return a version string.
std::string version();

/// Base class for bootstraps.
class Bootstrap {
 public:
  Bootstrap(){};
  virtual ~Bootstrap() = default;
  virtual int getRank() = 0;
  virtual int getNranks() = 0;
  virtual int getNranksPerNode() = 0;
  virtual void send(void* data, int size, int peer, int tag) = 0;
  virtual void recv(void* data, int size, int peer, int tag) = 0;
  virtual void allGather(void* allData, int size) = 0;
  virtual void barrier() = 0;

  void groupBarrier(const std::vector<int>& ranks);
  void send(const std::vector<char>& data, int peer, int tag);
  void recv(std::vector<char>& data, int peer, int tag);
};

/// A native implementation of the bootstrap using TCP sockets.
class TcpBootstrap : public Bootstrap {
 public:
  /// Create a random unique ID.
  /// @return The created unique ID.
  static UniqueId createUniqueId();

  /// Constructor.
  /// @param rank The rank of the process.
  /// @param nRanks The total number of ranks.
  TcpBootstrap(int rank, int nRanks);

  /// Destructor.
  ~TcpBootstrap();

  /// Return the unique ID stored in the @ref TcpBootstrap.
  /// @return The unique ID stored in the @ref TcpBootstrap.
  UniqueId getUniqueId() const;

  /// Initialize the @ref TcpBootstrap with a given unique ID.
  /// @param uniqueId The unique ID to initialize the @ref TcpBootstrap with.
  /// @param timeoutSec The connection timeout in seconds.
  void initialize(UniqueId uniqueId, int64_t timeoutSec = 30);

  /// Initialize the @ref TcpBootstrap with a string formatted as "ip:port" or "interface:ip:port".
  /// @param ifIpPortTrio The string formatted as "ip:port" or "interface:ip:port".
  /// @param timeoutSec The connection timeout in seconds.
  void initialize(const std::string& ifIpPortTrio, int64_t timeoutSec = 30);

  /// Return the rank of the process.
  int getRank() override;

  /// Return the total number of ranks.
  int getNranks() override;

  /// Return the total number of ranks per node.
  int getNranksPerNode() override;

  /// Send data to another process.
  ///
  /// Data sent via `send(senderBuff, size, receiverRank, tag)` can be received via `recv(receiverBuff, size,
  /// senderRank, tag)`.
  ///
  /// @param data The data to send.
  /// @param size The size of the data to send.
  /// @param peer The rank of the process to send the data to.
  /// @param tag The tag to send the data with.
  void send(void* data, int size, int peer, int tag) override;

  /// Receive data from another process.
  ///
  /// Data sent via `send(senderBuff, size, receiverRank, tag)` can be received via `recv(receiverBuff, size,
  /// senderRank, tag)`.
  ///
  /// @param data The buffer to write the received data to.
  /// @param size The size of the data to receive.
  /// @param peer The rank of the process to receive the data from.
  /// @param tag The tag to receive the data with.
  void recv(void* data, int size, int peer, int tag) override;

  /// Gather data from all processes.
  ///
  /// When called by rank `r`, this sends data from `allData[r * size]` to `allData[(r + 1) * size - 1]` to all other
  /// ranks. The data sent by rank `r` is received into `allData[r * size]` of other ranks.
  ///
  /// @param allData The buffer to write the received data to.
  /// @param size The size of the data each rank sends.
  void allGather(void* allData, int size) override;

  /// Synchronize all processes.
  void barrier() override;

 private:
  // The interal implementation.
  class Impl;

  // Pointer to the internal implementation.
  std::unique_ptr<Impl> pimpl_;
};

/// Enumerates the available transport types.
enum class Transport {
  Unknown,        // Unknown transport type.
  CudaIpc,        // CUDA IPC transport type.
  Nvls,           // NVLS transport type.
  IB0,            // InfiniBand device 0 transport type.
  IB1,            // InfiniBand device 1 transport type.
  IB2,            // InfiniBand device 2 transport type.
  IB3,            // InfiniBand device 3 transport type.
  IB4,            // InfiniBand device 4 transport type.
  IB5,            // InfiniBand device 5 transport type.
  IB6,            // InfiniBand device 6 transport type.
  IB7,            // InfiniBand device 7 transport type.
  Ethernet,       // Ethernet transport type.
  NumTransports,  // The number of transports.
};

const std::string TransportNames[] = {"UNK", "IPC", "NVLS", "IB0", "IB1", "IB2", "IB3",
                                      "IB4", "IB5", "IB6",  "IB7", "ETH", "NUM"};

namespace detail {
const size_t TransportFlagsSize = 12;
static_assert(TransportFlagsSize == static_cast<size_t>(Transport::NumTransports),
              "TransportFlagsSize must match the number of transports");
/// Bitset for storing transport flags.
using TransportFlagsBase = std::bitset<TransportFlagsSize>;
}  // namespace detail

/// Stores transport flags.
class TransportFlags : private detail::TransportFlagsBase {
 public:
  /// Default constructor for TransportFlags.
  TransportFlags() = default;

  /// Constructor for TransportFlags that takes a Transport enum value.
  ///
  /// @param transport The transport to set the flag for.
  TransportFlags(Transport transport);

  /// Check if a specific transport flag is set.
  ///
  /// @param transport The transport to check the flag for.
  /// @return True if the flag is set, false otherwise.
  bool has(Transport transport) const;

  /// Check if no transport flags are set.
  ///
  /// @return True if no flags are set, false otherwise.
  bool none() const;

  /// Check if any transport flags are set.
  ///
  /// @return True if any flags are set, false otherwise.
  bool any() const;

  /// Check if all transport flags are set.
  ///
  /// @return True if all flags are set, false otherwise.
  bool all() const;

  /// Get the number of transport flags that are set.
  ///
  /// @return The number of flags that are set.
  size_t count() const;

  /// Bitwise OR assignment operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the OR operation with.
  /// @return A reference to the modified TransportFlags.
  TransportFlags& operator|=(TransportFlags other);

  /// Bitwise OR operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the OR operation with.
  /// @return A new TransportFlags object with the result of the OR operation.
  TransportFlags operator|(TransportFlags other) const;

  /// Bitwise OR operator for TransportFlags and Transport.
  ///
  /// @param transport The Transport to perform the OR operation with.
  /// @return A new TransportFlags object with the result of the OR operation.
  TransportFlags operator|(Transport transport) const;

  /// Bitwise AND assignment operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the AND operation with.
  /// @return A reference to the modified TransportFlags.
  TransportFlags& operator&=(TransportFlags other);

  /// Bitwise AND operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the AND operation with.
  /// @return A new TransportFlags object with the result of the AND operation.
  TransportFlags operator&(TransportFlags other) const;

  /// Bitwise AND operator for TransportFlags and Transport.
  ///
  /// @param transport The Transport to perform the AND operation with.
  /// @return A new TransportFlags object with the result of the AND operation.
  TransportFlags operator&(Transport transport) const;

  /// Bitwise XOR assignment operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the XOR operation with.
  /// @return A reference to the modified TransportFlags.
  TransportFlags& operator^=(TransportFlags other);

  /// Bitwise XOR operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to perform the XOR operation with.
  /// @return A new TransportFlags object with the result of the XOR operation.
  TransportFlags operator^(TransportFlags other) const;

  /// Bitwise XOR operator for TransportFlags and Transport.
  ///
  /// @param transport The Transport to perform the XOR operation with.
  /// @return A new TransportFlags object with the result of the XOR operation.
  TransportFlags operator^(Transport transport) const;

  /// Bitwise NOT operator for TransportFlags.
  ///
  /// @return A new TransportFlags object with the result of the NOT operation.
  TransportFlags operator~() const;

  /// Equality comparison operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to compare with.
  /// @return True if the two TransportFlags objects are equal, false otherwise.
  bool operator==(TransportFlags other) const;

  /// Inequality comparison operator for TransportFlags.
  ///
  /// @param other The other TransportFlags to compare with.
  /// @return True if the two TransportFlags objects are not equal, false otherwise.
  bool operator!=(TransportFlags other) const;

  /// Convert the TransportFlags object to a bitset representation.
  ///
  /// @return A detail::TransportFlagsBase object representing the TransportFlags object.
  detail::TransportFlagsBase toBitset() const;

 private:
  /// Private constructor for TransportFlags that takes a bitset representation.
  ///
  /// @param bitset The bitset representation of the TransportFlags object.
  TransportFlags(detail::TransportFlagsBase bitset);
};

/// Bitwise OR operator for two Transport objects.
///
/// @param transport1 The first Transport to perform the OR operation with.
/// @param transport2 The second Transport to perform the OR operation with.
/// @return A new TransportFlags object with the result of the OR operation.
inline TransportFlags operator|(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) | transport2;
}

/// Bitwise AND operator for two Transport objects.
///
/// @param transport1 The first Transport to perform the AND operation with.
/// @param transport2 The second Transport to perform the AND operation with.
/// @return A new TransportFlags object with the result of the AND operation.
inline TransportFlags operator&(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) & transport2;
}

/// Bitwise XOR operator for two Transport objects.
///
/// @param transport1 The first Transport to perform the XOR operation with.
/// @param transport2 The second Transport to perform the XOR operation with.
/// @return A new TransportFlags object with the result of the XOR operation.
inline TransportFlags operator^(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) ^ transport2;
}

/// Get the number of available InfiniBand devices.
///
/// @return The number of available InfiniBand devices.
int getIBDeviceCount();

/// Get the name of the InfiniBand device associated with the specified transport.
///
/// @param ibTransport The InfiniBand transport to get the device name for.
/// @return The name of the InfiniBand device associated with the specified transport.
std::string getIBDeviceName(Transport ibTransport);

/// Get the InfiniBand transport associated with the specified device name.
///
/// @param ibDeviceName The name of the InfiniBand device to get the transport for.
/// @return The InfiniBand transport associated with the specified device name.
Transport getIBTransportByDeviceName(const std::string& ibDeviceName);

class Context;
class Connection;

/// Represents a block of memory that has been registered to a @ref Context.
class RegisteredMemory {
 public:
  /// Default constructor.
  RegisteredMemory() = default;

  /// Destructor.
  ~RegisteredMemory();

  /// Get a pointer to the memory block.
  ///
  /// @return A pointer to the memory block.
  void* data() const;

  /// Get a pointer to the original memory block.
  ///
  /// @return A pointer to the original memory block.
  void* originalDataPtr() const;

  /// Get the size of the memory block.
  ///
  /// @return The size of the memory block.
  size_t size();

  /// Get the transport flags associated with the memory block.
  ///
  /// @return The transport flags associated with the memory block.
  TransportFlags transports();

  /// Serialize the RegisteredMemory object to a vector of characters.
  ///
  /// @return A vector of characters representing the serialized RegisteredMemory object.
  std::vector<char> serialize();

  /// Deserialize a RegisteredMemory object from a vector of characters.
  ///
  /// @param data A vector of characters representing a serialized RegisteredMemory object.
  /// @return A deserialized RegisteredMemory object.
  static RegisteredMemory deserialize(const std::vector<char>& data);

 private:
  // The interal implementation.
  struct Impl;

  // Internal constructor.
  RegisteredMemory(std::shared_ptr<Impl> pimpl);

  // Pointer to the internal implementation. A shared_ptr is used since RegisteredMemory is immutable.
  std::shared_ptr<Impl> pimpl_;

  friend class Context;
  friend class Connection;
};

/// Represents one end of a connection.
class Endpoint {
 public:
  /// Default constructor.
  Endpoint() = default;

  /// Get the transport used.
  ///
  /// @return The transport used.
  Transport transport();

  /// Serialize the Endpoint object to a vector of characters.
  ///
  /// @return A vector of characters representing the serialized Endpoint object.
  std::vector<char> serialize();

  /// Deserialize a Endpoint object from a vector of characters.
  ///
  /// @param data A vector of characters representing a serialized Endpoint object.
  /// @return A deserialized Endpoint object.
  static Endpoint deserialize(const std::vector<char>& data);

 private:
  // The interal implementation.
  struct Impl;

  // Internal constructor.
  Endpoint(std::shared_ptr<Impl> pimpl);

  // Pointer to the internal implementation. A shared_ptr is used since Endpoint is immutable.
  std::shared_ptr<Impl> pimpl_;

  friend class Context;
  friend class Connection;
};

/// Represents a connection between two processes.
class Connection {
 public:
  virtual ~Connection() = default;

  /// Write data from a source @ref RegisteredMemory to a destination @ref RegisteredMemory.
  ///
  /// @param dst The destination @ref RegisteredMemory.
  /// @param dstOffset The offset in bytes from the start of the destination @ref RegisteredMemory.
  /// @param src The source @ref RegisteredMemory.
  /// @param srcOffset The offset in bytes from the start of the source @ref RegisteredMemory.
  /// @param size The number of bytes to write.
  virtual void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) = 0;

  /// Update a 8-byte value in a destination @ref RegisteredMemory and synchronize the change with the remote process.
  ///
  /// @param dst The destination @ref RegisteredMemory.
  /// @param dstOffset The offset in bytes from the start of the destination @ref RegisteredMemory.
  /// @param src A pointer to the value to update.
  /// @param newValue The new value to write.
  virtual void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) = 0;

  /// Flush any pending writes to the remote process.
  virtual void flush(int64_t timeoutUsec = 3e7) = 0;

  /// Get the transport used by the local process.
  ///
  /// @return The transport used by the local process.
  virtual Transport transport() = 0;

  /// Get the transport used by the remote process.
  ///
  /// @return The transport used by the remote process.
  virtual Transport remoteTransport() = 0;

  /// Get the name of the transport used for this connection
  ///
  /// @return name of @ref transport() -> @ref remoteTransport()
  std::string getTransportName();

 protected:
  // Internal methods for getting implementation pointers.
  static std::shared_ptr<RegisteredMemory::Impl> getImpl(RegisteredMemory& memory);
  static std::shared_ptr<Endpoint::Impl> getImpl(Endpoint& memory);
};

/// Used to configure an endpoint.
struct EndpointConfig {
  static const int DefaultMaxCqSize = 1024;
  static const int DefaultMaxCqPollNum = 1;
  static const int DefaultMaxSendWr = 8192;
  static const int DefaultMaxWrPerSend = 64;

  Transport transport;
  int ibMaxCqSize = DefaultMaxCqSize;
  int ibMaxCqPollNum = DefaultMaxCqPollNum;
  int ibMaxSendWr = DefaultMaxSendWr;
  int ibMaxWrPerSend = DefaultMaxWrPerSend;

  /// Default constructor. Sets transport to Transport::Unknown.
  EndpointConfig() : transport(Transport::Unknown) {}

  /// Constructor that takes a transport and sets the other fields to their default values.
  ///
  /// @param transport The transport to use.
  EndpointConfig(Transport transport) : transport(transport) {}
};

/// Represents a context for communication. This provides a low-level interface for forming connections in use-cases
/// where the process group abstraction offered by @ref Communicator is not suitable, e.g., ephemeral client-server
/// connections. Correct use of this class requires external synchronization when finalizing connections with the
/// @ref connect() method.
///
/// As an example, a client-server scenario where the server will write to the client might proceed as follows:
///   1. The client creates an endpoint with @ref createEndpoint() and sends it to the server.
///   2. The server receives the client endpoint, creates its own endpoint with @ref createEndpoint(), sends it to the
///      client, and creates a connection with @ref connect().
///   4. The client receives the server endpoint, creates a connection with @ref connect() and sends a
///      @ref RegisteredMemory to the server.
///   5. The server receives the @ref RegisteredMemory and writes to it using the previously created connection.
/// The client waiting to create a connection before sending the @ref RegisteredMemory ensures that the server can not
/// write to the @ref RegisteredMemory before the connection is established.
///
/// While some transports may have more relaxed implementation behavior, this should not be relied upon.
class Context {
 public:
  /// Create a context.
  Context();

  /// Destroy the context.
  ~Context();

  /// Register a region of GPU memory for use in this context.
  ///
  /// @param ptr Base pointer to the memory.
  /// @param size Size of the memory region in bytes.
  /// @param transports Transport flags.
  /// @return RegisteredMemory A handle to the buffer.
  RegisteredMemory registerMemory(void* ptr, size_t size, TransportFlags transports);

  /// Create an endpoint for establishing connections.
  ///
  /// @param config The configuration for the endpoint.
  /// @return The newly created endpoint.
  Endpoint createEndpoint(EndpointConfig config);

  /// Establish a connection between two endpoints. While this method immediately returns a connection object, the
  /// connection is only safe to use after the corresponding connection on the remote endpoint has been established.
  /// This method must be called on both endpoints to establish a connection.
  ///
  /// @param localEndpoint The local endpoint.
  /// @param remoteEndpoint The remote endpoint.
  /// @return std::shared_ptr<Connection> A shared pointer to the connection.
  std::shared_ptr<Connection> connect(Endpoint localEndpoint, Endpoint remoteEndpoint);

 private:
  // The interal implementation.
  struct Impl;

  // Pointer to the internal implementation.
  std::unique_ptr<Impl> pimpl_;

  friend class RegisteredMemory;
  friend class Endpoint;
};

/// A base class for objects that can be set up during @ref Communicator::setup().
struct Setuppable {
  virtual ~Setuppable() = default;

  /// Called inside @ref Communicator::setup() before any call to @ref endSetup() of any @ref Setuppable object that is
  /// being set up within the same @ref Communicator::setup() call.
  ///
  /// @param bootstrap A shared pointer to the bootstrap implementation.
  virtual void beginSetup(std::shared_ptr<Bootstrap> bootstrap);

  /// Called inside @ref Communicator::setup() after all calls to @ref beginSetup() of all @ref Setuppable objects that
  /// are being set up within the same @ref Communicator::setup() call.
  ///
  /// @param bootstrap A shared pointer to the bootstrap implementation.
  virtual void endSetup(std::shared_ptr<Bootstrap> bootstrap);
};

/// A non-blocking future that can be used to check if a value is ready and retrieve it.
template <typename T>
class NonblockingFuture {
  std::shared_future<T> future;

 public:
  /// Default constructor.
  NonblockingFuture() = default;

  /// Constructor that takes a shared future and moves it into the NonblockingFuture.
  ///
  /// @param future The shared future to move.
  NonblockingFuture(std::shared_future<T>&& future) : future(std::move(future)) {}

  /// Check if the value is ready to be retrieved.
  ///
  /// @return True if the value is ready, false otherwise.
  bool ready() const { return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

  /// Get the value.
  ///
  /// @return The value.
  ///
  /// @throws Error if the value is not ready.
  T get() const {
    if (!ready()) throw Error("NonblockingFuture::get() called before ready", ErrorCode::InvalidUsage);
    return future.get();
  }
};

/// A class that sets up all registered memories and connections between processes.
///
/// A typical way to use this class:
///   1. Call @ref connectOnSetup() to declare connections between the calling process with other processes.
///   2. Call @ref registerMemory() to register memory regions that will be used for communication.
///   3. Call @ref sendMemoryOnSetup() or @ref recvMemoryOnSetup() to send/receive registered memory regions to/from
///      other processes.
///   4. Call @ref setup() to set up all registered memories and connections declared in the previous steps.
///   5. Call @ref NonblockingFuture<RegisteredMemory>::get() to get the registered memory regions received from other
///      processes.
///   6. All done; use connections and registered memories to build channels.
///
class Communicator {
 public:
  /// Initializes the communicator with a given bootstrap implementation.
  ///
  /// @param bootstrap An implementation of the Bootstrap that the communicator will use.
  /// @param context An optional context to use for the communicator. If not provided, a new context will be created.
  Communicator(std::shared_ptr<Bootstrap> bootstrap, std::shared_ptr<Context> context = nullptr);

  /// Destroy the communicator.
  ~Communicator();

  /// Returns the bootstrap held by this communicator.
  ///
  /// @return std::shared_ptr<Bootstrap> The bootstrap held by this communicator.
  std::shared_ptr<Bootstrap> bootstrap();

  /// Returns the context held by this communicator.
  ///
  /// @return std::shared_ptr<Context> The context held by this communicator.
  std::shared_ptr<Context> context();

  /// Register a region of GPU memory for use in this communicator's context.
  ///
  /// @param ptr Base pointer to the memory.
  /// @param size Size of the memory region in bytes.
  /// @param transports Transport flags.
  /// @return RegisteredMemory A handle to the buffer.
  RegisteredMemory registerMemory(void* ptr, size_t size, TransportFlags transports);

  /// Send information of a registered memory to the remote side on setup.
  ///
  /// This function registers a send to a remote process that will happen by a following call of @ref setup(). The send
  /// will carry information about a registered memory on the local process.
  ///
  /// @param memory The registered memory buffer to send information about.
  /// @param remoteRank The rank of the remote process.
  /// @param tag The tag to use for identifying the send.
  void sendMemoryOnSetup(RegisteredMemory memory, int remoteRank, int tag);

  /// Receive memory on setup.
  ///
  /// This function registers a receive from a remote process that will happen by a following call of @ref setup(). The
  /// receive will carry information about a registered memory on the remote process.
  ///
  /// @param remoteRank The rank of the remote process.
  /// @param tag The tag to use for identifying the receive.
  /// @return NonblockingFuture<RegisteredMemory> A non-blocking future of registered memory.
  NonblockingFuture<RegisteredMemory> recvMemoryOnSetup(int remoteRank, int tag);

  /// Connect to a remote rank on setup.
  ///
  /// This function only prepares metadata for connection. The actual connection is made by a following call of
  /// @ref setup(). Note that this function is two-way and a connection from rank `i` to remote rank `j` needs
  /// to have a counterpart from rank `j` to rank `i`. Note that with IB, buffers are registered at a page level and if
  /// a buffer is spread through multiple pages and do not fully utilize all of them, IB's QP has to register for all
  /// involved pages. This potentially has security risks if the connection's accesses are given to a malicious process.
  ///
  /// @param remoteRank The rank of the remote process.
  /// @param tag The tag of the connection for identifying it.
  /// @param config The configuration for the local endpoint.
  /// @return NonblockingFuture<NonblockingFuture<std::shared_ptr<Connection>>> A non-blocking future of shared pointer
  /// to the connection.
  NonblockingFuture<std::shared_ptr<Connection>> connectOnSetup(int remoteRank, int tag, EndpointConfig localConfig);

  /// Get the remote rank a connection is connected to.
  ///
  /// @param connection The connection to get the remote rank for.
  /// @return The remote rank the connection is connected to.
  int remoteRankOf(const Connection& connection);

  /// Get the tag a connection was made with.
  ///
  /// @param connection The connection to get the tag for.
  /// @return The tag the connection was made with.
  int tagOf(const Connection& connection);

  /// Add a custom Setuppable object to a list of objects to be setup later, when @ref setup() is called.
  ///
  /// @param setuppable A shared pointer to the Setuppable object.
  void onSetup(std::shared_ptr<Setuppable> setuppable);

  /// Setup all objects that have registered for setup.
  ///
  /// This includes previous calls of @ref sendMemoryOnSetup(), @ref recvMemoryOnSetup(), @ref connectOnSetup(), and
  /// @ref onSetup(). It is allowed to call this function multiple times, where the n-th call will only setup objects
  /// that have been registered after the (n-1)-th call.
  void setup();

 private:
  // The interal implementation.
  struct Impl;

  // Pointer to the internal implementation.
  std::unique_ptr<Impl> pimpl_;
};

/// A constant TransportFlags object representing no transports.
extern const TransportFlags NoTransports;

/// A constant TransportFlags object representing all InfiniBand transports.
extern const TransportFlags AllIBTransports;

/// A constant TransportFlags object representing all transports.
extern const TransportFlags AllTransports;

/// A type which could be safely used in device side.
template <class T>
using DeviceHandle = typename T::DeviceHandle;

/// Retrieve the deviceHandle instance from host object.
template <typename T>
DeviceHandle<std::remove_reference_t<T>> deviceHandle(T&& t) {
  return t.deviceHandle();
}

/// Packet value type.
template <class T>
using PacketPayload = typename T::Payload;

}  // namespace mscclpp

namespace std {

/// Specialization of the std::hash template for mscclpp::TransportFlags.
template <>
struct hash<mscclpp::TransportFlags>;

}  // namespace std

#endif  // MSCCLPP_CORE_HPP_

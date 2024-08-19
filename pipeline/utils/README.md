## High-level Scheduler
> Note that this is implemented following design of [Punica](https://github.com/punica-ai/punica).
> CPU is used to form the metadata of batched KV-Cache, which is then fed into [FlashInfer](https://flashinfer.ai/) on GPU.
### Scheduling Policy
**First-Come-First-Serve (FCFS)**: The first job that arrives is the first to be scheduled.

### Benchmark
- [ShareGPT](): Syntheisze request distribution from the workload collection.
- Constant Batch: Feed infinite number of generation requests into the scheduler.
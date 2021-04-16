## Notes 2021-04-16
   1. ONNX in SGX from Microsoft - had a look, did not manage to run it, had issues; have a binary representation of protobuffer, causes issues; retry.
   2. Worked on the implementation of the target system - deploying a model, converting, etc:
    - running conversion in an own enclave; tf2onnx calls tensorflow, needs too much memory.
    - inference works; details to be ironed out;
   3. hypothesis: reduce number of system calls leading to OCALLS (mostly due to multi-threading); making the onnx run-time single-threaded AND removing syscalls causing OCALLS made the single-threaded version faster but it was still slower than the multi-threaded version.
  

## Notes 2021-03-26
   1. EPC size on Azure 26MB, 2MB; Thinkpad: 96 MB; why are there multiple sections? why on Azure? does this bring an advantage for virtualized deployments?
   2. Videos from OC3: https://youtube.com/playlist?list=PLEhAl3D5WVvSKTp9jD6lV67zuFR8lUZVa
   3. Microcode releases - new results, latest releases improve performance.
   4. OCALLS
   5. SGX Open source projects https://github.com/Maxul/Awesome-SGX-Open-Source
   6. ONNX in SGX from Microsoft: https://github.com/microsoft/onnx-server-openenclave
   7. Idea - compare performance of ONNX-on-SGX native and with a libraryOS; first step - try with existing VM, otherwise we'll look for alternatives.
   8. Getting TensorFlow working in ONNX

## Notes 2021-03-19
    1. Rewritten the benchmarking infrastructure to have better control;
    2. Tried different microcode releases - performance differences are limited (1,5% slower) - make sure to have a robust and reproducible benchmark;
    3. On Azure the performance impact is much larger - note as a result;
    4. Explore running only 1 thread to reduce number of OCALLs;
    5. Investigate reducing the EPC size;
    6. Do the conversion to include for timing;
    7. Investigate small batch sizes (1) with small pieces of data.

## Notes 2021-03-11
    1. Laptop - set up, progress.
    2. Azure VM - could ssh, all good;
    3. Client process/server process - inference outside and inside Occlum;
    4. Discussion about efficient IO: http://www.diva-portal.org/smash/get/diva2:1416013/FULLTEXT01.pdf
    5. Discussion about approaches to measurements; three factors: hardware/platform, library OS (or not)

## Notes 2021-03-05
    1. Discussion about Privado;
    2. Discussion about further steps:
      * Time to run ONNX in SGX
      * Is it worth protecting the architecture or is protecting the parameters enough?  
      * Implementing an architecture with discrete conversion Enclaves
      * Focus on **performance** first (potentially relevant paper: https://arxiv.org/pdf/2010.08440.pdf)
      * Focus on **side-channel attacks** second
    3. Arrange the SGX-enabled laptop and Azure VM.


## Notes 2021-02-26
    1. So far - literature review on SGX, overview of side-channels on memory
    access and existing solutions for SGX ML inference; library OS - differences
    and principles;
    2. Discussion about Privado: https://arxiv.org/pdf/1810.00602.pdf - nice find,
    gives some ideas about going forward.


## Notes 2021-02-12
    1. [DONE] List of literature - updated the readme; two main topics: SGX and libraryOS
    2. [DONE] High level time-plan;
    3. [ADMIN] Setting up the email and accessing the intranet;
    4. [ADMIN] Lenovo Laptop with SGX support and Azure VM will be provided when the time comes;
    5. Nicolae: Go through literature and potentially suggest more resources;

## Notes 2021-02-05
    1. [ADMIN] RISE Contract is DONE!
    2. [ADMIN] KTH examiner (RobertoG) and supervisor (DanielB) is confirmed.
    3. [TODO] List of literature (links for literature on LibraryOS available)
    5. [TODO] Create a timeplan for the thesis
    6. [THESIS] Discussion points:
        - benchmarking deltas: ONNX on different libOSs
        - choosing the right use case considering the onnx capabilities


## Notes 2021-01-29

1. [ADMIN] RISE Contract is in process
2. [ADMIN] Examiner is in process; Daniel Bosk will be the KTH supervisor.
3. [TODO] List of literature
4. [TODO] Create a Github repo and share with Nicolae
5. [TODO] Create a timeplan for the thesis
6. [THESIS] Discussion points:
    - Scope of the thesis (investigating different library OSs, benchmarking,
      security guarantees)
    - Approaches to converting from * to ONNX; effort of adding side-channel
    protection to one vs multiple runtimes;

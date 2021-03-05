## Notes 2021-03-05
    1. Discussion about Privado;
    2. Discussion about further steps:
      * Time to run ONNX in SGX
      * Is it worth protecting the architecture or is protecting the
        parameters enough?  
      * Implementing an architecture with discrete conversion Enclaves
      * Focus on **performance** first
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

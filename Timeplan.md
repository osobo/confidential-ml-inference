# Timeplan

## Until March 1
+ Do pre-study
+ Read previous work
+ Write short summaries of papers
+ Work on introduction and background section of report

## Until April 5
+ Based on knowledge gained: outline exact prototype(s) to implement and measurements to take
+ Start implementing, eg:
    + ONNX inside Occlum
    + Enclave that receives ONNX models and model inputs, and runs inference
    + Benchmarking
    + Conversion from TensorFlow to ONNX
    + More conversions
+ Write about method and implementation in report

## Until April 26
+ Identify bottlenecks, iterate on implementation. Could mean
    + Change how enclave uses ONNX runtime
    + Changes to Occlum
    + Changes to ONNX runtime
    + Use of different LibOS
+ Possibly run existing automated/standardized security evaluations
+ Write on all parts of report, especially method, result

## Until June 7
+ Analyse security of best iteration(s)
+ Identify what known attacks are feasible and their severity
+ Identify possible countermeasures, maybe implement
+ Finish report

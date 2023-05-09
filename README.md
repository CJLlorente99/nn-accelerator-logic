Logical steps of the proposed workflow.
1) Generation of the binarized neural network. See folders.
   - models. This follows an architecture based on the one defined in the paper "Energy-Efficient, Low-Latency Realization of Neural Networks through Boolean Logic Minimization"
   - modelsBNNPaper. This follows an architecture based on the one defined in the paper "Binarized Neural Networks: Training Neural Networks with Weights and Activations Contrained to +1 or -1"
2) Enumeration of the gradients and activations. To do so, use the following file from the importanceCalculation folder
   - gradientCalculation. Performs a sweep over the entire dataset to get the activations and gradients.
3) Calculation of the importance per neuron. To do so, use the following files from the importanceCalculation folder
    - legacyCalculation. Calculates the importance per neuron as proposed in paper "Interpret neural networks by identifying critical data routing paths".
    - gradientCalculation. Calculates the importance per neuron as proposed in paper "Neural response interpretation through the lens of critical pathways".
   Take into account that the legacy calculation is computationally expensive and, therefore, not recommendable for big NN.
4) Realization of the neurons as truth tables and assignation of importance per truth table entries. The goal of this step is to be able to further perform
minimization procedures (i.e. taking out the entries with no importance, which is equivalent to say taking out the entries that do not contribute to any class).
To do this, use the following file from the importanceCalculation folder.
   - ttGradientCalculation. Calculates the importance per neuron as proposed in paper "Neural response interpretation through the lens of critical pathways"
   and expands the concept to the truth table entries.
5) Perform a simplification of the truth tables based on importance values. To do so use the following file from the
layersTTOptimization folder.
    - mainOptimization.
6) Create the PLA files to be fed into the synthesis tools. To do so use the following file from the realization folder.
   - mainRealization. This file will create PLA files that can be used by the original ESPRESSO tool or by the ABC tool.
7) Creation and reduction of the boolean expression. Making use of ABC (AIG-based logic optimization) or ESPRESSO (SoP-based logic minimization)
the realization of the ISF represented in the PLA files can be generated. Additionally, output verilog and AIGER files can be generated.
8) Leveraging the output of the synthesis tools, these will be inputted into further-synthesis and mapping tools in order to generate a per layer
optimization (PLA files only, at least initially, define a neuron ISF). Thus, using the mapping utilities, the number of resources
will be quantified (in CLBs, LUTs or else).
9) In order to check accuracy of the definitive model, the AIGER files will be used to simulate each neurons output. This will done as follows:
   - A python script will recursively call a C++ file providing the AIGER file related to a particular neuron and the inputs to test in that neuron. The C++ file will generate the output per entry.
   - Another python script will compute the last layer operations (as it is binary-float) and compare it with the results. 
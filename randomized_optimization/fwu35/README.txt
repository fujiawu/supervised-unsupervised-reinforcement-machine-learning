Project 2: Randomized Optimization

CS-7641 Machine Learning
Spring 2015
Name: Fujia Wu
GT Account: fwu35

I used ABAGAIL for this assignment.
The evaluation functions for the optimization problems.
I created were written in java and built into ABAGAIL.jar.
The test codes were written in jython.

Steps to run my codes:

1. Download ABAGAIL: https://github.com/pushkar/ABAGAIL

2. Copy the following files to src/opt/example in ABAGAIL
KColorEvaluationFunction.java
KHillsEvaluationFunction.java
KnapsackEvaluationFunction.java
i.e., run this: cp *.java [path-to-ABAGAIL]/src/opt/example/
(The first 2 were written by me and the 3rd one is provided in ABAGAIL)

3. Build ABAGAIL with ant
i.e., run this: ant

4. Run the following jython codes:
jython -J-cp [path-to-ABAGAIL.jar] neuralnetwork.py < car.data
jython -J-cp [path-to-ABAGAIL.jar] khills.py
jython -J-cp [path-to-ABAGAIL.jar] kcolors.py
jython -J-cp [path-to-ABAGAIL.jar] knapsack.py
The neuralnetwork.py file runs the NN weights finding for data in "car.data" file
The khills.py file runs K-Hills optimization problem I created.
The kcolors.py file runs the K-Color problem for a random graph.
The knapsack.py file runs the KnapSack optimization problem.

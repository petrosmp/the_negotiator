An agent developed for the Multiagent Systems course in winter 2023.

Builds upon the techniques described and used in "Algorithm selection
in bilateral negotiation" by Litan Ilany and Ya’akov Gal.

Made to participate in a competition between TUC students.

We created "the_negotiator" agent which is a meta-agent that plays using pre-made strategies.

We modeled the online negotiation environment as a Multi-Armed-Bandit one using the UCB algorithm, just like in the paper and we 
aimed to give UCB a headstart by using Machine Learning.
Our approach is to use a Neural Network that we trained by testing our peer-designed agents on most of the given domains, and getting 
the returned utility along with selected domain features. 
Our dataset was created by collecting this data with a custom python script and was placed in the training_data folder.
We used the trainer.py script to pre-train the Neural Network model that outputs a preformance prediction for each strategy with one forward pass.
The neural network is then used to create some initial estimates for the different strategies when the negotiator agent sees the domain for the first time. 
Then the UCB algorithm takes over and plays the strategy with the highest confidence bound, adapting the confidence bounds according to the
utility that each played arm achieves.

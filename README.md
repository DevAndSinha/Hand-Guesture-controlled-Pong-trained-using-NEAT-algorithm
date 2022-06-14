# NEAT-Pong-Python
In this project, two very popular games – Ping-pong and Dino, have been automated using the NEAT algorithm. NeuroEvolution of Augmenting Topologies (NEAT) is a genetic algorithm (GA) for the generation of evolving artificial neural networks (a neuroevolution technique) developed by Ken Stanley in 2002 while at The University of Texas at Austin. We have further enhanced our project using Mediapipe and OpenCV for hand gesture control. MediaPipe is a Framework for building machine learning pipelines for processing time-series data like video, audio, etc. 
This cross-platform Framework works in Desktop/Server, Android, iOS, and embedded devices like Raspberry Pi and Jetson Nano. OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in commercial products.


Abstract:

NEAT (NeuroEvolution of Augmenting Topologies) is an evolutionary algorithm that creates artificial neural networks. To evolve a solution to a problem, the user must provide a fitness function which computes a single real number indicating the quality of an individual genome: better ability to solve the problem means a higher score. The algorithm progresses through a user-specified number of generations, with each generation being produced by reproduction (either sexual or asexual) and mutation of the most fit individuals of the previous generation. 
The reproduction and mutation operations may add nodes and/or connections to genomes, so as the algorithm proceeds, genomes (and the neural networks they produce) may become more and more complex. When the preset number of generations is reached, or when at least one individual (for a fitness criterion function of max; others are configurable) exceeds the user-specified fitness threshold, the algorithm terminates.
It uses a combination of the number of non-homologous nodes and connections with measures of how many homologous nodes and connections have diverged since their common origin. 
The NEAT algorithm chooses a direct encoding methodology because of this. Their representation is a little more complex than a simple graph or binary encoding, however, it is still straightforward to understand. It simply has two lists of genes, a series of nodes and a series of connections.
Using OpenCV, one can process images and videos to identify objects, faces, or even handwriting of a human. When integrated with various libraries, such as NumPy, python is capable of processing the OpenCV array structure for analysis.
MediaPipe provides cornerstone Machine Learning models for common tasks like hand tracking, therefore removing the same developmental bottleneck that exists for a host of Machine Learning applications. These models, along with their excessively easy-to-use APIs, in turn streamline the development process and reduce project lifetime for many applications that rely on Computer Vision.

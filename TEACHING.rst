***************************************
Notes on teaching this repository
***************************************

The intended audience has a working knowledge of Python, and a basic
understanding of statistics.  It is important to explain up front that
the materials are intended as both an introduction and reference so if
participants don't get everything then it is not critical.

If you are going to teach this workshop I would first watch this video.

    * https://www.youtube.com/watch?v=LlzVlqVzeD8

Then read through at least the first 3 chapters in Cam Davidison-Pilon's book to get some perspcetive

    * https://pymc-devs.github.io/pymc3/notebooks/getting_started.html
      
Then I would go through the getting started guide to get familiar with PyMC3
     
    * https://pymc-devs.github.io/pymc3/notebooks/getting_started.html

From these materials and from other sources a person who expects to teach these materials should be able to 

   * Conceptually explain MCMC
   * Talk about recent advances in Hamiltonian Monte Carlo methods like NUTS
   * Be able to talk about ADVI and mini-batch
   * Explain why using theano was a major step forward

Here is a reasonable way to break up the content.
     
   1. For about 1 hour go through the introductory examples
   2. Spend 5-10 minutes going through the recommender example (avoid a play-by-play)
   3. Spend 5-10 minutes going through the neural network example (avoid a play-by-play)


The introductory examples are the main contents and the recommender
and neural network examples are meant to showcase some of the more
advanced possibilities.

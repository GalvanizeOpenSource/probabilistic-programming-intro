*******************************************************
Introduction to probabilistics programming with PyMC3
*******************************************************

:Version: 1.0.0
:Authors: Adam Richards
:Web site: https://github.com/zipfian/probabilistic-programming-intro
:Copyright: This document has been placed in the public domain.
:License: These materials are released under the BSD-3 license unless otherwise noted

What is probabilistic programming all about?
-----------------------------------------------

There are three major trends in the machine learning side of data
science: **big data**, `deep learning <https://en.wikipedia.org/wiki/Deep_learning>`_ and **probabilistic
programming**.  There has been a sustained focus on the first two, but
recent advances in the way models are evaluated has brought the
attention again to probabilitstic programming.

At the core of probabilistic programming is the idea that statistical
models are written in code, which is then evaluated in turn by MCMC
sampling algorithms.  New variational inference algorithms have
emerged as a means to scale these algorithms to production level.

There are three major reasons to consider probabilistic programming.
With the Bayesian paradigm to guide model creation.

  1. **Customization** - We can create models that have built-in hypothesis tests
  2. **Propagation of uncertainty** - There is a degree of belief associated prediction and estimation
  3. **Intuition** - The models are essentially 'white-box' which provides insight into our data 
     
Overview
---------------------

Ultimately, we create models to guide the decision making process.  It
is a typical task in data science to build a recommendation system or
make a prediction about a medical diagnosis, but these recommendations
or diagnoses mean so much more when they are accompanied by an
estimated level of uncertainty.

This short talk is designed to familiarize Python programmers the basics
concepts of **probabilistic programming**.  We will introduce the
Python package PyMC3 with a tutorial example.  Then we will use it to
build a simple recommendation system and finally we will finish up
with an implementation of a probabilistic neural network.

Install
---------------

.. code:: bash
   
   pip install --process-dependency-links git+https://github.com/pymc-devs/pymc3

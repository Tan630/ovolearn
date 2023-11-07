Dear Dr. Kelly,

Thank you for reviewing this project!

Successes:
    * Onemax can be quickly optimized. You can see the result by runnning `python -m core.main` in `/src`.
        - I am once again convinced that onemax is a terrible problem to test against. Could you please recommend any alternatives?
    * The pendulum was successfully optimized before rewriting the framework broke it again.

Failures:
    * The gym testing environment is broken. You can see the result by runnning `python -m evolvables.expression` in `/src`.
        - The population becomes quickly overwhelmed by solutions that look the same. Might be because of the variator, which has a tendency to deposit duplicate solutions back to the population, or the variator which does not promote diversity. What are your recommendations?
        - The "pre-computed" best score of a solution is differnet from the actual performance of that solution. I am not quite sure how that can be fixed -- I am trying my hardest to find what went wrong. A similar problem was solved by making the scoring system accumulative, I am not sure if this can be solved in the same way.

The parent's selector must decide what to do if it cannot exactly fill parent tuples - for example, if pairs should be selected from a population of 11 genomes. What to do to the one that is left-over?

The crossover method worked before ... though I'm not sure if it contributes to the problem. There are too many moving pieces, which is taking up a lot of time to fix.


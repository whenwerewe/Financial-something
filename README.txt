This is a dynamic delta-hedger that takes in some timeseries of prices ("Underlying") over some period and returns the trades it would make in order to minimise exposure to the underlying
To use: either provide your own timeseries or use synthetic_crises.py to produce .csvs that resemble your chosen market conditions
(you may need to adjust the config to produce desired results)
Then run hedger.py to execute the simulated strategy and see a variety of diagnostics on how well it ran
For further information hedger_postproc.py gives more statistics and sensitivity_grid.py lets you evaluate the optimal choices of various parameters

.. torch-mvnorm documentation master file, created by
   sphinx-quickstart on Tue Aug 27 16:13:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Torch-MvNorm's documentation
========================================

.. automodule:: mvnorm

   .. data:: integration
   
      Controls the integration parameters:

               - ``integration.maxpts``, the maximum number of density evaluations (default 1000×d);
               - ``integration.abseps``, the absolute error tolerance (default 1e-6);
               - ``integration.releps``, the relative error tolerance (default 1e-6);
               - ``integration.n_jobs``, the number of jobs for ``joblib.Parallel`` (default to 1).


   .. autofunction:: multivariate_normal_cdf
   


.. toctree::




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

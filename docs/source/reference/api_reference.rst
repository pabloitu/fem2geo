API Documentation
=================


.. Here are the listed and linked the rst pages of the API docs. Hidden means it wont show on
.. this api reference landing page.

.. toctree::
   :maxdepth: 1
   :hidden:

   model
   tensor
   transform
   plots
   io
   schema

**Model**

.. currentmodule:: fem2geo.model

The :class:`~fem2geo.model.Model` class is the main handler of ``fem2geo``, which contains all
the model geometrical and result attributes. The class and its main methods are:

.. autosummary::
   :nosignatures:

    Model
    Model.stress

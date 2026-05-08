"""Custom modified variants of SOTA sequential models.

To register a new variant:

1. Drop a Python module here, e.g. ``sasrec_plus.py``::

       from recbole.model.sequential_recommender.sasrec import SASRec

       class SASRecPlus(SASRec):
           # override forward / loss / etc. as needed
           ...

2. In ``src/models/__init__.py``, import the class and add an entry to
   ``MODEL_REGISTRY``::

       from models.variants.sasrec_plus import SASRecPlus

       MODEL_REGISTRY["SASRecPlus"] = {
           "class": SASRecPlus,
           "type": "sequential",
           "search_space": sasrec_space,   # reuse parent space, or define a new one
           "static": _seq_static(),
       }

3. Re-run the benchmark notebook. Resumability ensures only the new model trains.
"""

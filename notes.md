Understanding the pipeline tree of the pipeline.(gets the scaler object from the leaf node.)

pipe                                    # this is a Pipeline
pipe.named_steps['preprocesor']         # ✓ Pipeline → use named_steps
                                         # → returns the ColumnTransformer object

preprocesor_obj = pipe.named_steps['preprocesor']   # now I'm holding a ColumnTransformer
preprocesor_obj.named_transformers_['num']          # ✓ ColumnTransformer → use named_transformers_
                                                      # → returns the numeric_pipeline (a Pipeline again!)

numeric_pipe_obj = preprocesor_obj.named_transformers_['num']   # now holding a Pipeline again
numeric_pipe_obj.named_steps['scaler']                          # ✓ Pipeline → back to named_steps


Actual code to implement:(example to get the median value from the leaf pipeline step)
lr_pipeline.named_steps['preprocesor'].named_transformers_['num'].named_steps['imputer'].statistics_
def binary_mapper(series, mapping , col):
    clean = series.str.lower().str.strip()
    mapped = clean.map(mapping)
    bad = clean.notna() & mapped.isna() # here notna() is opposite of isnull(), anding both of them gives the one which are bad values, i.e is there value? , is the result missing/ Nan? if both are true it's bad./// also & is for series anding normally 'and' is used.
    #also , bad is a column of true or false like a mask of binaries/series of binariess
    if bad.any():  #at least one is true in the list/series
        raise ValueError(f"{col}: Unrecognized {clean[bad].unique().tolist()}; expected {list(mapping)}") #clean[bad] lines up 'clean' and 'bad' row by row and keep the true values only
    return mapped

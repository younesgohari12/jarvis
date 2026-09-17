def numeric_context(text,span,family):
    a,b=span
    return 'domain '+family+' LEFT '+text[max(0,a-60):a]+' NUMBER RIGHT '+text[b:b+60]

import json
from core import fit_centroids,classify,fit_line
from features import mse
with open('examples/input.json') as stream: data=json.load(stream)
model=fit_centroids(data['rows'],data['labels'])
line=fit_line(data['x'],data['y'])
print(json.dumps({'class':classify(model,[1,1]),'line':line,
    'training_mse':mse(data['y'],[line['slope']*x+line['intercept'] for x in data['x']])},sort_keys=True))

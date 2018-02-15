var CONFIG = {
  "geojson": "data/countries_states.geojson",
  "yearMs": 120000,
  "calendar": {
    "marker": "#marker"
  },
  "colorKey": {
    "el": "#color-key-canvas",
    "gradient": "data/colorGradientRainbow.json"
  },
  "globes": [
    {
      "el": '#ocean-currents',
      "title": "Ocean surface currents and<br />sea surface temperature",
      "video": "#video",
      "viewAngle": 45,
      "near": 0.01,
      "far": 1000,
      "radius": 0.5,
      "minMag": 0.33,
      "precision": 0.01,
      "animationMs": 3000,
      "geojsonLineColor": 0x555555
    },{
      "el": '#atmosphere-wind',
      "title": "Wind and temperature<br />10 meters above sea level",
      "video": "#video",
      "videoOffset": 0.5,
      "viewAngle": 45,
      "near": 0.01,
      "far": 1000,
      "radius": 0.5,
      "minMag": 0.33,
      "precision": 0.01,
      "animationMs": 3000,
      "geojsonLineColor": 0x443820
    }
  ]
};

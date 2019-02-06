# lod-wave

## Current status:
#### Broken due to missing methods, for example `convertpIndexToCoordinate` in `gridlod.util`.
#### This is lieky due to the Python 3 refactor

## Installing dependencies
Simply run `pip install -r requirements.txt` to install most of the dependencies.

The one exception is the `scikit-sparse` package, which was broken when this was tested.

This needs to be installed with `pip install scikit-sparse`.

`scikit-sparse` additionally requires `libsuitesparse-dev` to be installed. You can do this by running:
```
sudo apt install libsuitesparse-dev
```


## Updating dependencies
Most Python dependencies are managed by `pip-compile`, so you will need that tool.

To add a new package, simply add it to the list in `requirements.in`.

You then update the `requirements.txt`-file by running
````
pip-compile --output-file requirements.txt requirements.in
```
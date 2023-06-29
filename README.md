# wavaetro
wavelet based explorations of the MAESTRO Dataset


## Project desription

### The dataset

The [MAESTRO Dataset v 3.00](https://magenta.tensorflow.org/datasets/maestro#v300) is a pair of mid


## Usage

### Env setup

```bash
git clone https://github.com/zar3bski/wavaetro.git
cd wavaetro

poetry install
```

### Get the dataset

```bash
mkdir -p data
cd data

wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip

# Skip this if you are not paranoid
echo "6680fea5be2339ea15091a249fbd70e49551246ddbd5ca50f1b2352c08c95291  maestro-v3.0.0.zip" | sha256sum --check
echo "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c  maestro-v3.0.0-midi.zip" | sha256sum --check

unzip maestro-v3.0.0
```
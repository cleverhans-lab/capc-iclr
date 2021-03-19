# CAPC

Confidential And Private Collaborative Machine Learning

We develop a protocol involving a small number of parties (e.g., a few hospitals) who want to improve the utility of their respective models via collaboration, and a third party Content Service Provider (CSP). Each hospital first trains its own model on its local dataset. We assume all parties. The CSP generates a pair of secret and public keys for an additive homomorphic encryption scheme and sends the public key to all collaborating parties. Once a party identified a query they would like a label for, they initiate the collaboration protocol.


Install he_transformer from here: https://github.com/IntelAI/he-transformer. We use the ubuntu 18. version.

First, install the crypto packages.
wget https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py
python install.py -install -tool -ot -sh2pc

Then, make the cpp code:
cd gc-emp-test
cmake . && make

Make sure libtmux is installed.

train a cryptonets-relu.pb model and store it in ~/models/cryptonets-relu.pb

Open a tmux pane named "capc"
tmux new -s capc
create 3 panes, and in pane 0 run: 

python run_experiment.py --n_parties=X --ignore_parties --encryption_params=config/10.json

replace X with the number of desired parties.
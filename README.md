# CAPC Learning

Confidential And Private Collaborative Machine Learning

Paper: https://openreview.net/forum?id=h2EbJ4_wMVq

Intuition: We develop a protocol involving a small number of parties (e.g., a few hospitals) who want to improve the utility of their respective models via collaboration. Each party trains its own model on its local dataset. We assume all parties. Once a party identifies a query they would like a label for, they initiate the collaboration protocol.

### Full abstract: 
Abstract: Machine learning benefits from large training datasets, which may not always be possible to collect by any single entity, especially when using privacy-sensitive data. In many contexts, such as healthcare and finance, separate parties may wish to collaborate and learn from each other's data but are prevented from doing so due to privacy regulations. Some regulations prevent explicit sharing of data between parties by joining datasets in a central location (confidentiality). Others also limit implicit sharing of data, e.g., through model predictions (privacy). There is currently no method that enables machine learning in such a setting, where both confidentiality and privacy need to be preserved, to prevent both explicit and implicit sharing of data. Federated learning only provides confidentiality, not privacy, since gradients shared still contain private information. Differentially private learning assumes unreasonably large datasets. Furthermore, both of these learning paradigms produce a central model whose architecture was previously agreed upon by all parties rather than enabling collaborative learning where each party learns and improves their own local model. We introduce Confidential and Private Collaborative (CaPC) learning, the first method provably achieving both confidentiality and privacy in a collaborative setting. We leverage secure multi-party computation (MPC), homomorphic encryption (HE), and other techniques in combination with privately aggregated teacher models. We demonstrate how CaPC allows participants to collaborate without having to explicitly join their training sets or train a central model. Each party is able to improve the accuracy and fairness of their model, even in settings where each party has a model that performs well on their own dataset or when datasets are not IID and model architectures are heterogeneous across parties. 

### Code:
Install he_transformer from here: https://github.com/IntelAI/he-transformer. We use Ubuntu 18.

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

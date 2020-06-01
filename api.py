import sys
import os
import json
import shutil
import traceback
import time
from flask import Flask, request, jsonify
import pandas as pd
import pickle


app = Flask(__name__)

# inputs
training_data = 'data/data_clean.csv'
item_dict = 'data/auxiliary/item_dict.json'
user_dict = 'data/auxiliary/user_dict.json'
interactions = 'data/auxiliary/interactions.csv'

model_directory = 'model'
auxiliary_directory = 'data/auxiliary'
model_file_name = f'{model_directory}/recommender.pkl'

# variables to populate during training
recommender = None

@app.route('/items_to_user', methods=['POST', 'GET']) # Create http://host:port/items_to_user POST end point
def items_to_user():
    if request.method == 'POST':
        if recommender:
            try:
                from utils import items_to_user

                interactions_ = pd.read_csv(interactions)
                interactions_.set_index('CustomerID', inplace=True)
                with open(user_dict) as f:
                    user_dict_ = json.loads(f.read())
                with open(item_dict) as f:
                    item_dict_ = json.loads(f.read())

                json_ = request.json #capture the json from POST
                user_id = int(json_["user_id"])
                nrec_items = int(json_["nrec_items"])
                show_known = bool(json_["show_known"])
                recs = items_to_user(model=recommender, interactions=interactions_, user_id=user_id,
                                        user_dict=user_dict_, item_dict=item_dict_, threshold=0, nrec_items=nrec_items, show_known=show_known)
                return jsonify(recs)
            except Exception as e:
                return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    elif request.method == 'GET':
        return 'Please send a POST-request with args to get recommendations'

@app.route('/users_to_item', methods=['POST']) # Create http://host:port/users_to_item POST end point
def users_to_item():
    if request.method == 'POST':
        if recommender:
            try:
                from utils import users_to_item

                interactions_ = pd.read_csv(interactions)
                interactions_.set_index('CustomerID', inplace=True)
                with open(user_dict) as f:
                    user_dict_ = json.loads(f.read())
                with open(item_dict) as f:
                    item_dict_ = json.loads(f.read())

                json_ = request.json #capture the json from POST
                item_id = json_["item_id"]
                len_users = int(json_["len_users"])
                users_for_item = {}
                recs = users_to_item(model = recommender, interactions = interactions_, item_id = item_id,
                                    user_dict = user_dict_,
                                    item_dict = item_dict_,
                                    len_users = len_users)
                users_for_item[item_id] = recs
                return jsonify(users_for_item)
            except Exception as e:
                return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    elif request.method == 'GET':
        return 'Please send a POST-request with args to get recommendations'

@app.route('/items_to_item', methods=['POST']) # Create http://host:port/items_to_item POST end point
def items_to_item():
    if request.method == 'POST':
        if recommender:
            try:
                from utils import items_to_item, create_item_emdedding_distance_matrix

                interactions_ = pd.read_csv(interactions)
                interactions_.set_index('CustomerID', inplace=True)
                with open(item_dict) as f:
                    item_dict_ = json.loads(f.read())

                json_ = request.json #capture the json from POST
                item_id = json_["item_id"]
                n_items = int(json_["n_items"])
                
                item_item_dist = create_item_emdedding_distance_matrix(model = recommender, interactions = interactions_)
                recs = items_to_item(item_emdedding_distance_matrix = item_item_dist,
                                        item_id = item_id,
                                        item_dict = item_dict_,
                                        n_items = n_items)
                
                return jsonify(recs)
            except Exception as e:
                return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    elif request.method == 'GET':
        return 'Please send a POST-request with args to get recommendations'

@app.route('/train', methods=['GET']) # Create http://host:port/train GET end point
def train():

    start = time.time()

    from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
    from utils import clean_data, to_sparse, create_user_dict, create_item_dict, fit_mf_model
    from utils import items_to_user, items_to_item, create_item_emdedding_distance_matrix, users_to_item
    print('Modules loaded...')
    training_metrics={}

    data = pd.read_csv(training_data)
    piv, cols, interactions_ = to_sparse(data)
    interactions_.to_csv(interactions, index=True)

    user_dict_ = create_user_dict(interactions=interactions_)
    item_dict_ = create_item_dict(df = data, id_col = 'StockCode', name_col = 'Description')
    
    with open(user_dict, 'w') as json_file:
        json.dump(user_dict_, json_file)
    with open(item_dict, 'w') as json_file:
        json.dump(item_dict_, json_file)
    
    print('Data preparations ready...')
    mf_model = fit_mf_model(interactions = interactions_,
                            n_components = 140,
                            loss = 'warp',
                            epoch = 10,
                            n_jobs = 6)
    print('Model fit...')
    training_metrics["precision_at_3"] = round(precision_at_k(mf_model, piv, k=3).mean()*100)
    training_metrics["recall_at_3"] = round(recall_at_k(mf_model, piv, k=3).mean()*100)
    training_metrics["auc_score"]=round(auc_score(mf_model, piv).mean()*100)

    pickle.dump(mf_model, open(str(model_directory + "/" +"recomender.pkl"), "wb"))
    print('Model trained & serialized in %.1f seconds' % (time.time() - start))
    
    return jsonify(training_metrics)

@app.route('/wipe', methods=['GET']) # Create http://host:port/wipe GET end point
def wipe():
    try:
        shutil.rmtree('model')
        shutil.rmtree('data/auxiliary')
        os.makedirs(model_directory)
        os.makedirs(auxiliary_directory)
        return 'Model deleted, auxiliary intermediate data wiped!'

    except Exception as e:
        print(str(e))
        return 'Unable to delete and recreate model directory'



if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        with open(str(model_directory + "/" + "recomender.pkl"), "rb") as file:
            recommender = pickle.load(file)
        print('Recommender loaded...')


    except Exception as e:
        print('No recommedner model found...')
        print('Train first!')
        print(str(e))
        recommender = None

    app.run(host='0.0.0.0', port=port, debug=True)


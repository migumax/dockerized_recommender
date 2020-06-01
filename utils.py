import datetime as dt
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import auc_score

def clean_data(data):
    '''
    Clean original data using a set of empirical rules.
    Arguments:
    - data: original transactional df
    Output:
    - data: cleaned df
    '''
    data.InvoiceDate = pd.to_datetime(data.InvoiceDate, format="%m/%d/%Y %H:%M")
    data = data[data["UnitPrice"] >= 0 ]
    data = data[data["InvoiceNo"].astype(str).str[0] != "C"]
    data = data[data["InvoiceNo"].astype(str).str[0] != "A"]
    data = data[data["Quantity"] > 0 ]
    data["Description"] = data["Description"].fillna("Unkown")
    data["CustomerID"] = data["CustomerID"].fillna(-9999)
    data['Year'] = data.InvoiceDate.dt.year
    data['Revenue'] = data['Quantity']*data['UnitPrice']
    data = data[data["CustomerID"]!=-9999]

    # we specifically only leave data for the UK and 2011
    data = data[data['Country'] == 'United Kingdom']
    data = data[data['Year'] == 2011]

    return data

def to_sparse(data):
    '''
    Create sparse intercations matrix from the original transactional dataset.
    Arguments:
    - data: transactional df with the following required columns: 'StockCode', 'CustomerID', 'Revenue'/'Price'
    Output:
    - piv: csr-type interactions pivot table of shape n_users*m_items
    - cols: column names of piv
    - interactions: interactions pivot table of shape n_users*m_items
    '''
    d = data[['StockCode', 'CustomerID', 'Revenue']].groupby(['StockCode', 'CustomerID']).count()#[1:10000]
    d = d.reset_index()
    piv = pd.pivot_table(d, index='CustomerID', columns='StockCode', values='Revenue')
    piv[piv>=1]=1
    piv = piv.dropna(axis=1, how='all')
    cols = piv.columns
    piv = piv.fillna(0)
    interactions= piv.copy()
    piv = lil_matrix(piv, dtype='float')
    return piv, cols, interactions

def create_user_dict(interactions):
    '''
    Create a user dictionary based on their index and id in interactions df
    Arguments: 
        - interactions: dataframe of shape n_users*m_items containing transactional history 
    Output:
       - user_dict - standard python dict of type {"user_id": index}
    '''
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0
    for i in user_id:
        user_dict[round(i)] = counter
        counter += 1
    return user_dict


def create_item_dict(df, id_col, name_col):
    '''
    Create an item dictionary based on their item_id and item_name
    Arguments: 
        - df: dataframe with items data
        - id_col: column name containing unique identifier for an item
        - name_col: column name containing name of the item
    Output:
        - item_dict = standard python dict of type {"item_id": item_name}
    '''
    item_dict = {}
    for i in range(df.shape[0]):
        item_dict[(df.at[df.index[i], id_col])] = df.at[df.index[i], name_col]
    return item_dict


def fit_mf_model(interactions, n_components, loss='warp', epoch=3, n_jobs=6):
    '''
    Create csr matrix out of interactions df, create and fit Matrix Factorization model.
    Find more about parameters on LightFM's official docs page:
    https://making.lyst.com/lightfm/docs/lightfm.html
    Arguments:
        - interactions: dataframe of shape n_users*m_items containing transactional history 
        - n_components:  the dimensionality of the feature latent embeddings
        - loss:  one of (‘logistic’, ‘bpr’, ‘warp’, ‘warp-kos’): the loss function
        - epoch: number of epochs to run 
        - n_jobs: number of cores used for running the training process
    Output:
        - model object which is an instance of LightFM class
    '''
    X = csr_matrix(interactions.values)
    model = LightFM(no_components=n_components, loss=loss)
    model.fit(X, epochs=epoch, num_threads=n_jobs)
    return model


def items_to_user(model, interactions, user_id, user_dict,
                            item_dict, threshold=0, nrec_items=10, show_known=False):
    '''
    Create recommendations for 1 user.
    Arguments:
        - model: model object of LightFM class
        - interactions: dataframe of shape n_users*m_items containing transactional history
        - user_id 
        - user_dict 
        - item_dict
        - threshold: value above which the rating is favorable in new interaction matrix
        - nrec_items: number of items to recommend 
    Output: 
        - List of items the given user has already bought
        - List of nrec_items which user is likely to be interested in
    '''
    recommendations = {}
    n_users, n_items = interactions.shape
    user_x = user_dict[str(user_id)]
    scores = pd.Series(model.predict(user_x, np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))

    known_items = list(pd.Series(interactions.loc[user_id, :]
                                [interactions.loc[user_id, :] > threshold].index).sort_values(ascending=False))

    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    recommendations["user_id"] = str(user_id)
    recommendations["recs_ids"] = return_score_list
    recommendations["recs"] = scores
    if show_known == True:
        recommendations["known"]=known_items
    return recommendations


def users_to_item(model, interactions, item_id, user_dict, item_dict, len_users):
    '''
    Create a list of top N interested users for a given item
    Arguments:
        - model: model object of LightFM class
        - interactions: dataframe of shape n_users*m_items containing transactional history
        - item_id 
        - user_dict 
        - item_dict
        - len_users: number of users needed as an output
    Output:
        - user_list: list of recommended users 
    '''
    n_users, n_items = interactions.shape
    x = np.array(interactions.columns)
    scores = pd.Series(model.predict(np.arange(n_users),
                                     np.repeat(x.searchsorted(item_id), n_users)))
    user_list = list(interactions.index[scores.sort_values(
        ascending=False).head(len_users).index])
    return [int(x) for x in user_list]


def create_item_emdedding_distance_matrix(model, interactions):
    '''
    Create item-item distance embedding matrix
    Arguments:
        - model: model object of LightFM class
        - interactions: dataframe of shape n_users*m_items containing transactional history
    Output:
        - item_emdedding_distance_matrix: dataframe containing pair-wise cosine distance matrix between items
    '''
    df_item_norm_sparse = csr_matrix(model.item_embeddings)
    similarities = cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = interactions.columns
    item_emdedding_distance_matrix.index = interactions.columns
    return item_emdedding_distance_matrix


def items_to_item(item_emdedding_distance_matrix, item_id,
                             item_dict, n_items=10):
    '''
    Function to create item-item recommendation
    Arguments:
        - item_emdedding_distance_matrix: dataframe containing pair-wise cosine distance matrix between items
        - item_id
        - item_dict
        - n_items: number of items needed as an output
    Output:
        - recommended_items: list of recommended items
    '''
    recommendations = {}
    recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id, :].
                                       sort_values(ascending=False).head(n_items+1).
                                       index[1:n_items+1]))
    recommendations["item_id"] = item_id
    recommendations["item"] = item_dict[item_id]
    recommendations["recs_ids"] = recommended_items
    recommendations["recs"] = [item_dict[i] for i in recommended_items]
    return recommendations
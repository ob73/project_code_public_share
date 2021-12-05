import random
import pickle
import os
import numpy as np

import pandas as pd


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.opponent_number = 1 - agent_number  # index for opponent
        self.n_items = params["n_items"]

        # Unpickle the trained model
        # Complications: pickle should work with any machine learning models
        # However, this does not work with custom defined classes, due to the way pickle operates
        # TODO you can replace this with your own model
        print(os.getcwd())
        self.filename = 'yourteamname_files/trained_model_p1'
        self.trained_model = pickle.load(open(self.filename, 'rb'))
        
        self.knn_test = pickle.load(open('data/knn_test_p1', 'rb'))
        self.item0_embedding = pickle.load(open('data/item0embedding', 'rb'))
        self.item1_embedding = pickle.load(open('data/item1embedding', 'rb'))
        self.test_covariates = pickle.load(open('data/test_covariate_compatible', 'rb'))
        self.test_embeddings = pickle.load(open('data/test_noisy_embedding_compatible', 'rb'))
        self.test_covariate_with_index = self.test_covariate.reset_index()
        self.user_ids_of_users_with_embeddings = set(self.test_noisy_embedding.reset_index()['index'])
        self.covariates_of_test_users_with_vectors = self.test_covariate_with_index[self.test_covariate_with_index['index'].isin(self.user_ids_of_users_with_embeddings)]
        self.covariates_of_test_users_without_vectors = self.test_covariate_with_index[~self.test_covariate_with_index['index'].isin(self.user_ids_of_users_with_embeddings)]
        self.covariates_only_of_test_users_without_vectors = self.covariates_of_test_users_without_vectors.drop('index', axis=1)

    def _process_last_sale(self, last_sale, profit_each_team):
        # print("last_sale: ", last_sale)
        # print("profit_each_team: ", profit_each_team)
        my_current_profit = profit_each_team[self.this_agent_number]
        opponent_current_profit = profit_each_team[self.opponent_number]

        my_last_prices = last_sale[2][self.this_agent_number]
        opponent_last_prices = last_sale[2][self.opponent_number]

        did_customer_buy_from_me = last_sale[1] == self.this_agent_number
        did_customer_buy_from_opponent = last_sale[1] == self.opponent_number

        which_item_customer_bought = last_sale[0]

        # print("My current profit: ", my_current_profit)
        # print("Opponent current profit: ", opponent_current_profit)
        # print("My last prices: ", my_last_prices)
        # print("Opponent last prices: ", opponent_last_prices)
        # print("Did customer buy from me: ", did_customer_buy_from_me)
        # print("Did customer buy from opponent: ",
        #       did_customer_buy_from_opponent)
        # print("Which item customer bought: ", which_item_customer_bought)

        # TODO - add your code here to potentially update your pricing strategy based on what happened in the last round
        pass

    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items=2, indicating prices this agent is posting for each item.
    def action(self, obs):
        new_buyer_covariates, new_buyer_embedding, last_sale, profit_each_team = obs
        if new_buyer_embedding is None:
            most_similar_user_id = self.knn_test.kneighbors(self.covariates_only_of_test_users_without_vectors.iloc[i].to_numpy().reshape(1,-1), 1, return_distance=False)
            most_similar_user = self.covariates_of_test_users_with_vectors.iloc[most_similar_user_id[0][0]]
            new_buyer_embedding = self.test_noisy_embedding.loc[most_similar_user['index']]
        
        item0_dotproduct = np.dot(new_buyer_embedding, self.item0_embedding)
        item1_dotproduct = np.dot(new_buyer_embedding, self.item1_embedding)
        
        item0_max_price = 2.2222207173210085    
        item1_max_price = 3.9996456192290166
        item0_prices = np.linspace(0,item0_max_price,1000)
        item1_prices = np.linspace(0,item1_max_price,1000)
        np.random.seed(0)
        item0_prices_sampled = np.random.choice(item0_prices,10)
        item1_prices_sampled = np.random.choice(item1_prices,10)
        price_pairs = []
        for item_0_price in item0_prices_sampled:
            for item_1_price in item1_prices_sampled:
                price_pairs.append((item_0_price, item_1_price))

        max_revenue_for_test_individual = 0 
        max_price_pair_for_test_individual = ()
        for price_pair in price_pairs:
            test_features_with_price_individual = list(price_pair) + list(new_buyer_covariates) + [item0_dotproduct, item1_dotproduct]
            demand_function = self.trained_model.predict_proba(np.asarray(test_features_with_price_individual).reshape(1, -1)) # probabilities according to classes {-1,0,1}
            expected_revenue_for_price_pair = demand_function[0][1] * price_pair[0] + demand_function[0][2] * price_pair[1]
            if expected_revenue_for_price_pair > max_revenue_for_test_individual:
                max_revenue_for_test_individual = expected_revenue_for_price_pair
                max_price_pair_for_test_individual = price_pair


        self._process_last_sale(last_sale, profit_each_team)
        # return self.trained_model.predict(np.array([1, 2, 3]).reshape(1, -1))[0] + random.random()
        return max_price_pair_for_test_individual
        # TODO Currently this output is just a deterministic 2-d array, but the students are expected to use the buyer covariates to make a better prediction
        # and to use the history of prices from each team in order to create prices for each item.

import torch
# from torch_geometric.data import Data
import numpy as np

class stock_state():
    def __init__(self, stock_name, stock_type="stock", num_of_stocks=0, stock_value=0, lots=[]):
        self.stock_name    = stock_name    # noqa
        self.stock_type    = stock_type    # noqa
        self.num_of_stocks = num_of_stocks # noqa
        self.stock_value   = stock_value   # noqa
        self.total_value   = num_of_stocks*stock_value # noqa
        self.lots          = lots # noqa
        self.profit_tax    = 0.25 # noqa

        lot = {'amount': num_of_stocks, 'value': stock_value}
        self.lots.append(lot)

    @property
    def num_stocks(self):
        return self.num_of_stocks

    @property
    def stock_val(self):
        return self.stock_value

    def print_stock_stats(self):
        return f"{self.stock_name}:\n# of stocks: {self.num_of_stocks :.2f}\nstock value: {self.stock_value}\n"

    def update_stock_value(self, stock_value):
        self.stock_value = stock_value
        return

    def buy_stocks(self, money):
        """
        :param money: how much money the agent invested in stocks
        :return: leftovers: money left from buying INTEGERS amount of stocks
        description: buy amount of stocks matching to money and stock value. create lot and push to lot list
        """
        value     = self.stock_value    # noqa
        amount    = int(money / value)  # noqa
        leftovers = np.mod(money, value)

        self.num_of_stocks += amount

        if self.stock_type == "money":
            return 0
        self.total_value = self.num_of_stocks * value

        lot = {'amount': amount, 'value': value}
        self.lots.append(lot)
        return leftovers

    def sell_stocks(self, amount):
        """
        :param amount: number of stocks to sell
        :return: selling value/amount: total amount of money made from the sell
        description: sell stocks from lots list, from latest lot to newest
        """
        assert amount <= self.num_of_stocks, "error of amount in sell"
        self.num_of_stocks -= amount

        if self.stock_type == "money":
            return amount

        value          = self.stock_value # noqa
        selling_value  = 0 # noqa
        selling_amount = 0
        while amount > 0:
            lot = self.lots[0]
            if amount > lot['amount']:
                selling_amount = lot['amount']
                self.lots.pop(0)
            else:
                selling_amount = amount
                self.lots[0]['amount'] -= selling_amount

            selling_value += selling_amount*(value - self.profit_tax*(value-lot['value']))
            amount -= selling_amount
        return selling_value

    def stock_reward(self):
        """
        :return: reward: total value of the stocks the agent hold
        """
        if self.stock_type=="money":
            return self.num_of_stocks

        profit_tax = self.profit_tax
        value      = self.stock_value # noqa
        reward     = 0                # noqa
        for lot in self.lots:
            reward += lot['amount'] * value  # - profit_tax*(value-lot['value'])
        return reward

class env():
    def __init__(self, init_money=5000, init_UPRO_val=24.64, init_UPRO_amount=500, init_TMF_val=21.65, init_TMF_amount=500, num_of_actions=19): # noqa
        self.Money_Node = stock_state(stock_name="money", stock_type="money", num_of_stocks=init_money,       stock_value=1,             lots=[]) # noqa
        self.UPRO_Node  = stock_state(stock_name="UPRO",  stock_type="stock", num_of_stocks=init_UPRO_amount, stock_value=init_UPRO_val, lots=[]) # noqa
        self.TMF_Node   = stock_state(stock_name="TMF",   stock_type="stock", num_of_stocks=init_TMF_amount,  stock_value=init_TMF_val,  lots=[]) # noqa

        self.investing_steps = {0: 0.05, 1: 0.1, 2: 0.15}
        # self.target_ratio    = (init_UPRO_amount*init_UPRO_val) / (init_TMF_amount*init_TMF_val)
        self.target_ratio    = (init_UPRO_amount) / (init_TMF_amount) # noqa
        self.threshold       = 0.3 * self.target_ratio # noqa
        self.num_of_actions  = num_of_actions  # noqa

    @property
    def stocks_ratio(self):
        return self.UPRO_Node.num_stocks / self.TMF_Node.num_stocks
        # return (self.UPRO_Node.num_stocks*self.UPRO_Node.stock_val) / (self.TMF_Node.num_stocks*self.TMF_Node.stock_val)

    @property
    def ratio_distance(self):
        ratio     = self.stocks_ratio # noqa
        target    = self.target_ratio # noqa
        threshold = self.threshold # noqa
        distance  = (1 - self.is_in_ratio())* (abs(ratio - target) - threshold) # noqa
        # make sure distance is in (0,1) range
        if distance > 1:
            distance = 1 - 1/distance

        return distance

    def update_stock_values(self, UPRO_val, TMF_val):
        self.UPRO_Node.update_stock_value(UPRO_val)
        self.TMF_Node.update_stock_value(TMF_val)
        return

    def observation(self):  # cur weights should be in size: 3X2
        """
        :return: numpy 3x2 Features matrix, each row contains: stock_value, stock_amount
        """
        Connection_mat = np.array([[0, 1 / 2, 1 / 2], [1 / 2, 0, 1 / 2], [1 / 2, 1 / 2, 0]])  # 3X3
        f_UPRO       = np.array([self.UPRO_Node.stock_val/10, self.UPRO_Node.num_stocks/10]) # noqa
        f_TMF        = np.array([self.TMF_Node.stock_val/10,  self.TMF_Node.num_stocks/10]) # noqa
        f_money      = np.array([1, self.Money_Node.num_stocks/100]) # noqa
        Features_mat = np.vstack([f_UPRO, f_TMF, f_money])  # noqa - 3X2
        Features_mat = np.matmul(Connection_mat, Features_mat) # noqa
        Features_mat = Features_mat.reshape([-1, ]) # noqa
        Features_mat = np.append(Features_mat, self.ratio_distance) # noqa
        return Features_mat
        # return np.matmul(Connection_mat, Features_mat)  # output is 3X2

    def reward(self):
        return (self.TMF_Node.stock_reward() + self.UPRO_Node.stock_reward() + self.Money_Node.stock_reward()) * (1 - self.ratio_distance) / 100 # noqa

    def step(self, action):
        """"
        input:
            action - integer between 0 to 18
        output:
            observation - new state observation
            reward      - reward from selected action
        description
            do action on enviorment:
                edge is selected by int(action/6) - 0-2
                direction is selected by mod(action,6) - over/under 3, if true: money->upro->tmf->money
                investment is selected by mod(action,3) - according to env dictionary
        """
        if action == self.num_of_actions:
            return self.observation(), self.reward()
        investing_percentage = self.investing_steps[np.mod(action, 3)]
        edge      = int(action/6) # noqa
        direction = np.mod(action, 6) < 3
        if edge == 0:  # noqa - UPRO-money
            if direction:
                amount        = investing_percentage * self.Money_Node.num_stocks # noqa
                selling_value = self.Money_Node.sell_stocks(amount)               # noqa
                leftovers     = self.UPRO_Node.buy_stocks(selling_value)          # noqa
                self.Money_Node.buy_stocks(leftovers)
            else:
                amount        = int(investing_percentage * self.UPRO_Node.num_stocks) # noqa
                selling_value = self.UPRO_Node.sell_stocks(amount)
                self.Money_Node.buy_stocks(selling_value)

        elif edge == 1:  # money-TMF
            if direction:
                amount        = int(investing_percentage * self.TMF_Node.num_stocks) # noqa
                selling_value = self.TMF_Node.sell_stocks(amount)
                self.Money_Node.buy_stocks(selling_value)
            else:
                amount        = investing_percentage * self.Money_Node.num_stocks # noqa
                selling_value = self.Money_Node.sell_stocks(amount)
                leftovers     = self.TMF_Node.buy_stocks(selling_value) # noqa
                self.Money_Node.buy_stocks(leftovers)

        else:  # noqa - UPRO-TMF
            if direction:
                amount        = int(investing_percentage * self.UPRO_Node.num_stocks) # noqa
                selling_value = self.UPRO_Node.sell_stocks(amount)
                leftovers     = self.TMF_Node.buy_stocks(selling_value) # noqa
                self.Money_Node.buy_stocks(leftovers)
            else:
                amount        = int(investing_percentage * self.TMF_Node.num_stocks) # noqa
                selling_value = self.TMF_Node.sell_stocks(amount)
                leftovers     = self.UPRO_Node.buy_stocks(selling_value) # noqa
                self.Money_Node.buy_stocks(leftovers)

        return self.observation(), self.reward()

    def is_in_ratio(self):
        # return True
        tr = self.target_ratio
        th = self.threshold
        # return (tr - th < (self.UPRO_Node.num_stocks*self.UPRO_Node.stock_val) / (self.TMF_Node.num_stocks*self.TMF_Node.stock_val) < tr + th)
        return (tr - th < self.stocks_ratio < tr + th)

    def choose_action_for_ratio(self):
        """
        :return: action: which action to choose (between 12 to 17)
        description: compute the current ratio vs. target ratio, and choose action to set the diff smaller
        """
        ratio = self.stocks_ratio

        if ratio > self.target_ratio:  # noqaq - move UPRO to TMF
            action = 12
        else:                          # noqa - Move TMF to UPRO
            action = 15

        return action

    def print_env(self):
        return f"{self.Money_Node.print_stock_stats()}{self.UPRO_Node.print_stock_stats()}{self.TMF_Node.print_stock_stats()}"

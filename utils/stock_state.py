import torch
from torch_geometric.data import Data
import numpy as np

class stock_state():
    def __init__(self, stock_name, stock_type="stock", num_of_stocks=0, stock_value=0, lots=[]):
        self.stock_name    = stock_name
        self.stock_type    = stock_type
        self.num_of_stocks = num_of_stocks
        self.stock_value   = stock_value
        self.total_value   = num_of_stocks*stock_value
        self.lots          = lots
        self.profit_tax    = 0.25

    @property
    def num_stocks(self):
        return self.num_of_stocks

    @property
    def stock_val(self):
        return self.stock_value

    def update_stock_value(self, stock_value):
        self.stock_value = stock_value
        return

    def buy_stocks(self, money):
        """
        :param money: how much money the agent invested in stocks
        :return: buy amount of stocks matching to money and stock value. create lot and push to lot list
        """
        value  = self.stock_value
        amount = money / value
        self.num_of_stocks += amount
        if (self.stock_type == "money"):
            return
        self.total_value = self.num_of_stocks * value

        lot = {'amount': amount, 'value': value}
        self.lots.append(lot)
        return

    def sell_stocks(self, amount):
        """
        :param amount: number of stocks to sell
        :return: selling value/amount: total amount of money made from the sell
        description: sell stocks from lots list, from latest lot to newest
        """
        assert amount <= self.num_of_stocks, "error of amount in sell"
        if self.stock_type=="money":
            self.num_of_stocks -= amount
            return amount
        value          = self.stock_value
        selling_value  = 0
        selling_amount = 0
        while amount > 0:
            lot = self.lots[0]
            if amount > lot['amount']:
                selling_amount = lot['amount']
                self.lots.pop(0)
            else:
                selling_amount = amount
                self.lots[0]['amount'] -= selling_amount

            selling_value += selling_amount*(value - self.profit_tax*np.max(0, value-lot['value']))
            amount -= selling_amount
        return selling_value

    def stock_reward(self):
        """
        :return: reward: total value of the stocks the agent hold
        """
        if self.stock_type=="money":
            return self.num_of_stocks

        profit_tax = self.profit_tax
        value      = self.stock_value
        reward     = 0
        for lot in self.lots:
            reward += lot['amount'] * (value - profit_tax*np.max(0, value-lot['value']))
        return reward

class state():
    def __init__(self, init_money=10000, init_UPRO_val=42.75, init_UPRO_amount=10, init_TMF_val=24.93, init_TMF_amount=10):
        self.Money_Node = stock_state(stock_name="money", stock_type="money", num_of_stocks=init_money,       stock_value=1,             lots=[])
        self.UPRO_Node  = stock_state(stock_name="UPRO",  stock_type="stock", num_of_stocks=init_UPRO_amount, stock_value=init_UPRO_val, lots=[])
        self.TMF_Node   = stock_state(stock_name="TMF",   stock_type="stock", num_of_stocks=init_TMF_amount,  stock_value=init_TMF_val,  lots=[])

        self.investing_steps ={0: 0.1, 1: 0.2, 2: 0.3}

    def update_stock_values(self, UPRO_val, TMF_val):
        self.UPRO_Node.update_stock_value(UPRO_val)
        self.TMF_Node.update_stock_value(TMF_val)
        return

    def state_observation(self):  # cur weights should be in size: 3X2
        """
        :return: numpy 3x2 Features matrix, each row contains: stock_value, stock_amount
        """
        Connection_mat = torch.tensor([[0, 1 / 2, 1 / 2], [1 / 2, 0, 1 / 2], [1 / 2, 1 / 2, 0]])  # 3X3
        f_UPRO  = np.array([self.UPRO_Node.stock_val, self.UPRO_Node.num_stocks])
        f_TMF   = np.array([self.TMF_Node.stock_val,  self.TMF_Node.num_stocks])
        f_money = np.array([1, self.Money_Node.num_stocks])
        Features_mat = np.vstack([f_UPRO, f_TMF, f_money])  # 3X2
        # return Features_mat
        return np.matmul(Connection_mat, Features_mat)  # output is 3X2

    def reward(self):
        return self.TMF_Node.stock_reward() + self.UPRO_Node.stock_reward() + self.Money_Node.stock_reward()

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
        if action == 18:
            return self.state_observation(), self.reward()
        investing_percentage = self.investing_steps[np.mod(action, 3)]
        edge      = int(action/6)
        direction = np.mod(action, 6) < 3
        if edge == 0:  # UPRO-money
            if direction:
                amount = investing_percentage * self.Money_Node.num_stocks
                selling_value = self.Money_Node.sell_stocks(amount, 1)
                self.UPRO_Node.buy_stocks(selling_value)
            else:
                amount = investing_percentage * self.UPRO_Node.num_stocks
                selling_value = self.UPRO_Node.sell_stocks(amount)
                self.Money_Node.buy_stocks(selling_value)

        elif edge == 1:  # money-TMF
            if direction:
                amount = investing_percentage * self.TMF_Node.num_stocks
                selling_value = self.TMF_Node.sell_stocks(amount)
                self.Money_Node.buy_stocks(selling_value)
            else:
                amount = investing_percentage * self.Money_Node.num_stocks
                selling_value = self.Money_Node.sell_stocks(amount, 1)
                self.TMF_Node.buy_stocks(selling_value)

        else:  # UPRO-TMF
            if direction:
                amount = investing_percentage * self.UPRO_Node.num_stocks
                selling_value = self.UPRO_Node.sell_stocks(amount)
                self.TMF_Node.buy_stocks(selling_value)
            else:
                amount = investing_percentage * self.TMF_Node.num_stocks
                selling_value = self.TMF_Node.sell_stocks(amount)
                self.UPRO_Node.buy_stocks(selling_value)

        return self.state_observation(), self.reward()






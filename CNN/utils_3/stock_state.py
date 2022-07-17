import torch
from torch_geometric.data import Data
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

        value      = self.stock_value # noqa
        reward     = 0                # noqa
        for lot in self.lots:
            reward += lot['amount'] * value  # - profit_tax*(value-lot['value'])
        return reward

class env():
    def __init__(self, init_money=5000, init_UPRO_val=24.64, init_UPRO_amount=500, init_TMF_val=21.65, init_TMF_amount=500, num_of_actions=19, use_ratio=1): # noqa
        self.Money_Node = stock_state(stock_name="money", stock_type="money", num_of_stocks=init_money,       stock_value=1,             lots=[]) # noqa
        self.UPRO_Node  = stock_state(stock_name="UPRO",  stock_type="stock", num_of_stocks=init_UPRO_amount, stock_value=init_UPRO_val, lots=[]) # noqa
        self.TMF_Node   = stock_state(stock_name="TMF",   stock_type="stock", num_of_stocks=init_TMF_amount,  stock_value=init_TMF_val,  lots=[]) # noqa

        # self.edge_index = torch.tensor([[0, 1],
        #                            [1, 0],
        #                            [1, 2],
        #                            [2, 1],
        #                            [2, 0],
        #                            [0, 2]], dtype=torch.long)

        self.investing_steps = {0: 0.05, 1: 0.1, 2: 0.15}
        self.target_ratio      = self.stocks_ratio # noqa
        self.threshold       = 0.3 * self.target_ratio # noqa
        self.num_of_actions  = num_of_actions  # noqa
        self.use_ratio       = use_ratio # noqa
        self.last_reward     = self.reward() # noqa

    @property
    def stocks_ratio(self):
        # return self.UPRO_Node.num_stocks / self.TMF_Node.num_stocks
        return (self.UPRO_Node.num_stocks*self.UPRO_Node.stock_val) / (self.TMF_Node.num_stocks*self.TMF_Node.stock_val)

    @property
    def ratio_distance(self):
        ratio     = self.stocks_ratio # noqa
        target    = self.target_ratio # noqa
        threshold = self.threshold # noqa
        distance  = (1 - self.is_in_ratio())* (abs(ratio - target) - threshold) # noqa
        # make sure distance is in (0,1) range
        if distance > 1:
            distance = 1 - 1/distance

        return distance * self.use_ratio

    def update_stock_values(self, UPRO_val, TMF_val):
        self.UPRO_Node.update_stock_value(UPRO_val)
        self.TMF_Node.update_stock_value(TMF_val)
        return

    def observation(self):  # cur weights should be in size: 3X2
        """
        :return: graph of 3 nodes (each node has its features) , ratio distance
        """
        # [UPRO-val, UPRO-am, TMF-val, TMF-am, Money-val, Money-am, connected2UPRO, connected2TMF, connected2MONEY]
        x = [[self.UPRO_Node.stock_val/10, self.UPRO_Node.num_stocks/100, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, self.TMF_Node.stock_val,  self.TMF_Node.num_stocks/100, 0, 0, 1, 0, 1],
             [0, 0, 0, 0, 1, self.Money_Node.num_stocks/1000, 1, 1, 0]]


        for upro_lot in self.UPRO_Node.lots:
            x.append([upro_lot['value'] / 10, upro_lot['amount'] / 100, 0, 0, 0, 0, 1, 0, 0])

        for tmf_lot in self.TMF_Node.lots:
            x.append([0, 0, tmf_lot['value'], tmf_lot['amount'] / 100, 0, 0, 0, 1, 0])

        graph = torch.tensor(x, dtype=torch.float)
        return graph

    def reward(self, final_iteration=False):
        reward = (self.TMF_Node.stock_reward() + self.UPRO_Node.stock_reward() + self.Money_Node.stock_reward()) / 100 # noqa
        if final_iteration:
            return reward
        return reward * (1 - self.ratio_distance)

    def step(self, action):
        """"
        input:
            action - integer between 0 to 18
        output:
            observation - new state observation
            reward      - reward from selected action
        description
            do action on enviorment:
                node is selected by int(action/3) - 0-2
                direction is selected by mod(action,2)
                0 - with arrow
                1 - against arrow
                UPRO(0) -> TMF(1) -> Money(2)
        """
        last_reward = self.reward()
        edge      = int(action/3) # noqa
        direction = np.mod(action, 2)
        if edge == 0:  # noqa - UPRO
            if direction == 0:
                investing_percentage = self.cal_selling_amount(sell_stock="UPRO", buy_stock="TMF")
                amount = int(investing_percentage * self.UPRO_Node.num_stocks)  # noqa
                selling_value = self.UPRO_Node.sell_stocks(amount)
                leftovers = self.TMF_Node.buy_stocks(selling_value)  # noqa
                self.Money_Node.buy_stocks(leftovers)
            else:
                investing_percentage = self.cal_selling_amount(sell_stock="UPRO", buy_stock="money")
                amount = int(investing_percentage * self.UPRO_Node.num_stocks)  # noqa
                selling_value = self.UPRO_Node.sell_stocks(amount)
                self.Money_Node.buy_stocks(selling_value)

        elif edge == 1:  # TMF
            if direction == 0:
                investing_percentage = self.cal_selling_amount(sell_stock="TMF", buy_stock="money")
                amount = int(investing_percentage * self.TMF_Node.num_stocks)  # noqa
                selling_value = self.TMF_Node.sell_stocks(amount)
                self.Money_Node.buy_stocks(selling_value)
            else:
                investing_percentage = self.cal_selling_amount(sell_stock="TMF", buy_stock="UPRO")
                amount = int(investing_percentage * self.TMF_Node.num_stocks)  # noqa
                selling_value = self.TMF_Node.sell_stocks(amount)
                leftovers = self.UPRO_Node.buy_stocks(selling_value)  # noqa
                self.Money_Node.buy_stocks(leftovers)

        else:  # noqa Money
            if direction == 0:
                investing_percentage = self.cal_selling_amount(sell_stock="money", buy_stock="UPRO")
                amount = investing_percentage * self.Money_Node.num_stocks  # noqa
                selling_value = self.Money_Node.sell_stocks(amount)
                leftovers = self.UPRO_Node.buy_stocks(selling_value)  # noqa
                self.Money_Node.buy_stocks(leftovers)
            else:
                investing_percentage = self.cal_selling_amount(sell_stock="money", buy_stock="TMF")
                amount = investing_percentage * self.Money_Node.num_stocks  # noqa
                selling_value = self.Money_Node.sell_stocks(amount)
                leftovers = self.TMF_Node.buy_stocks(selling_value)  # noqa
                self.Money_Node.buy_stocks(leftovers)
        self.last_reward = self.reward()
        return self.observation(), self.reward() - last_reward

    def is_in_ratio(self):
        # return True
        tr = self.target_ratio
        th = self.threshold
        return tr - th < self.stocks_ratio < tr + th

    def cal_selling_amount(self, sell_stock: str="", buy_stock: str=""):
        stock_percentage = 0.05
        amount = 0
        need_to_sell_upro = (self.stocks_ratio > self.target_ratio) and not self.is_in_ratio()
        need_to_sell_tmf = (self.stocks_ratio < self.target_ratio) and not self.is_in_ratio()

        if (sell_stock == "TMF" and self.TMF_Node.num_stocks == 0) or (sell_stock == "UPRO" and self.UPRO_Node.num_stocks == 0) \
            or (sell_stock == "money" and self.Money_Node.num_stocks == 0):
            return 0
        # 3 situations where ratio can't be reached!
        if need_to_sell_upro and (sell_stock == "TMF" or buy_stock == "UPRO"):
            amount = stock_percentage

        elif need_to_sell_tmf and (sell_stock == "UPRO" or buy_stock == "TMF"):
            amount = stock_percentage

        elif self.is_in_ratio():
            amount = stock_percentage

        elif sell_stock == "money":
            if buy_stock == "TMF":
                amount = self.UPRO_Node.num_stocks*self.UPRO_Node.stock_val / self.target_ratio - self.TMF_Node.num_stocks*self.TMF_Node.stock_val
                amount = amount / self.Money_Node.num_stocks
            if buy_stock == "UPRO":
                amount = self.target_ratio*self.TMF_Node.num_stocks*self.TMF_Node.stock_val - self.UPRO_Node.stock_val*self.UPRO_Node.num_stocks
                amount = amount / self.Money_Node.num_stocks

        elif buy_stock == "money":
            if sell_stock == "TMF":
                amount = self.TMF_Node.num_stocks - self.UPRO_Node.num_stocks * self.UPRO_Node.stock_val / (self.target_ratio*self.TMF_Node.stock_val)
                amount = np.ceil(amount).astype(int) / self.TMF_Node.num_stocks
            if sell_stock == "UPRO":
                amount = self.UPRO_Node.num_stocks - self.target_ratio * self.TMF_Node.num_stocks * self.TMF_Node.stock_val / self.UPRO_Node.stock_val
                amount = np.ceil(amount).astype(int) / self.UPRO_Node.num_stocks

        # return num_stocks-1 as it was the best ratio distance
        elif sell_stock == "UPRO":
            ratio_dis = np.inf
            for num_stocks in range(self.UPRO_Node.num_stocks):
                new_TMF  = self.TMF_Node.num_stocks + num_stocks*self.UPRO_Node.stock_val / self.TMF_Node.stock_value
                new_UPRO = self.UPRO_Node.num_stocks - num_stocks
                new_ratio = (new_UPRO*self.UPRO_Node.stock_val) / (new_TMF*self.TMF_Node.stock_val)
                if abs(new_ratio - self.target_ratio) > ratio_dis:
                    amount = (num_stocks-1) / self.UPRO_Node.num_stocks
                    break
                ratio_dis = abs(new_ratio - self.target_ratio)
                amount = num_stocks / self.UPRO_Node.num_stocks

        elif sell_stock == "TMF":
            ratio_dis = np.inf
            for num_stocks in range(self.TMF_Node.num_stocks):
                new_UPRO = self.UPRO_Node.num_stocks + num_stocks * self.TMF_Node.stock_val / self.UPRO_Node.stock_value
                new_TMF  = self.TMF_Node.num_stocks - num_stocks
                new_ratio = (new_UPRO * self.UPRO_Node.stock_val) / (new_TMF * self.TMF_Node.stock_val)
                if abs(new_ratio - self.target_ratio) > ratio_dis:
                    amount = (num_stocks-1) / self.TMF_Node.num_stocks
                    break
                ratio_dis = abs(new_ratio - self.target_ratio)
                amount = num_stocks / self.TMF_Node.num_stocks
        return min(amount, 1)







    def print_env(self):
        msg =   "stock name | stock value | stock amount\n" # noqa
        msg += f"money      | 1           | {int(self.Money_Node.num_of_stocks)}\n"
        msg += f"UPRO       | {self.UPRO_Node.stock_value:.2f}       | {self.UPRO_Node.num_of_stocks}\n"
        msg += f"TMF        | {self.TMF_Node.stock_value:.2f}       | {self.TMF_Node.num_of_stocks}\n"
        return msg



        # elif sell_stock == "UPRO":
        #     num_stocks =  np.arange(1, self.UPRO_Node.num_stocks + 1)
        #     new_TMF  = self.TMF_Node.num_stocks + num_stocks*self.UPRO_Node.stock_val / self.TMF_Node.stock_value
        #     new_UPRO = self.UPRO_Node.num_stocks - num_stocks
        #     new_ratio = (new_UPRO*self.UPRO_Node.stock_val) / (new_TMF*self.TMF_Node.stock_val)
        #     ratio_dis = np.abs(new_ratio - self.target_ratio)
        #     amount = (np.argmax(ratio_dis) + 1) / self.UPRO_Node.num_stocks
        #
        #
        # elif sell_stock == "TMF":
        #     num_stocks = np.arange(1, self.TMF_Node.num_stocks)
        #     new_UPRO  = self.UPRO_Node.num_stocks + num_stocks * self.TMF_Node.stock_val / self.UPRO_Node.stock_value
        #     new_TMF   = self.TMF_Node.num_stocks - num_stocks
        #     new_ratio = (new_UPRO * self.UPRO_Node.stock_val) / (new_TMF * self.TMF_Node.stock_val)
        #     ratio_dis = np.abs(new_ratio - self.target_ratio)
        #     amount = (np.argmax(ratio_dis) + 1)/ self.TMF_Node.num_stocks























import torch
from torch_geometric.data import Data

class stock_state():
    def __init__(self, stock_name, stock_type="stock", num_of_stocks=0, stock_value=0, total_value=0, lots=[]):
        self.stock_name    = stock_name
        self.stock_type    = stock_type
        self.num_of_stocks = num_of_stocks
        self.stock_value   = stock_value
        self.total_value   = total_value
        self.lots          = lots

    def buy_stocks(self, amount, value, date):
        self.num_of_stocks += amount
        if (self.stock_type == "money"):
            return
        self.total_value   = self.num_of_stocks * value
        self.stock_value   = value

        lot = {'amount': amount, 'value': value, 'date': date}
        self.lots.append(lot)

    def sell_stocks(self, amount, value):
        assert amount <= self.num_of_stocks, "error of amount in sell"
        if self.stock_type=="money":
            self.num_of_stocks -= amount
            return amount
        self.stock_value = value
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

            selling_value += selling_amount*value
            amount -= selling_amount
        return selling_value

# class state():
#     edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
#                                [1, 0, 2, 1, 2, 0]], dtype=torch.long)
#     nodes = torch.tensor([[stock_state("UPRO", num_of_stocks=0, stock_value=0, total_value=0, lots=[]), ]])
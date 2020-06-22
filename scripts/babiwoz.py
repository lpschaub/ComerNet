"""

Code pour convertir un dialogue DSTC en dialogue multiwoz. 
Deux modes disponibles : full babi ou full multiwoz. 

"""
import glob
import json, os
from pprint import pprint


class Turn(object):
    def __init__(self, turndic, systemdic, turn_idx, domain):
        self.turndic = turndic
        self.systemdic = systemdic
        self.turn_label = []
        self.turn_idx = turn_idx
        self.domain = domain.replace('[', "").replace(']', "").replace("'", "")
        self.output = ""
        self.belief = []
        self.usr = ""
        self.system_acts = []

    def fill_infos(self, belief_tracker):
        Turn.get_label(self)
        Turn.get_system(self)
        Turn.get_usr(self)
        Turn.get_belief(self, belief_tracker=belief_tracker)
        Turn.get_system_acts(self)

    def get_label(self):
        self.turn_label = [[key, value] for key, value in self.turndic["goal-labels"].items()]
        # print(self.turn_label)

    def get_system(self):
        self.output = self.systemdic['output']['transcript']
        # print(self.output)

    def get_usr(self):
        self.usr = self.turndic['transcription']
        # print(self.usr)

    def get_belief(self, belief_tracker):
        for elem in self.turndic['semantics']['json']:
            belief_tracker.append(elem)
            for dic in elem:
                if dic == 'slots':

                    stack = belief_tracker[-1]
                    if stack[dic] and elem[dic]:
                        if stack[dic][0][0] == elem[dic][0][0]:
                            if stack[dic][0][1] != elem[dic][0][1]:
                                belief_tracker[-1][dic][0][1] = elem[dic][0][1]

            # belief_tracker.append()
        self.belief = [elem for elem in self.turndic["semantics"]["json"]]
        for elem in belief_tracker :
            if elem not in self.belief :
                self.belief.append(elem)
        # print(self.belief)
        # print("\n\nand the other :: \n\n")
        # self.belief = belief_tracker
        # print(belief_tracker)
        # print(self.belief)

    def get_system_acts(self):
        self.system_acts = [value for k in self.systemdic['output']['dialog-acts'] for value in k['slots']]
        # print(self.system_acts)


class Converter(object):
    def __init__(self, dialjson, systemjson, mode='woz', domain='restaurant'):
        self.domains = [domain]
        self.dialog_idx = ""
        self.dial = json.load(open(dialjson, encoding='utf-8'))
        # pprint(self.dial)
        self.turns = []
        self.system = json.load(open(systemjson, encoding='utf-8'))
        self.belief_tracker = []

    def getkeys(self):
        self.keys = [elem for elem in self.dial]
        # print(self.keys)

    def getid(self):
        self.dialog_idx = self.dial['caller-id']
        # print(self.dialog_idx)

    def parse_turns(self, id=0):
        if not self.dial['turns']:
            # for turn in self.turns :
            #     print(turn.belief)
            return 0
        else:
            new_turn = Turn(self.dial['turns'].pop(0), self.system['turns'].pop(0), id,
                            str(self.domains))
            new_turn.fill_infos(self.belief_tracker)
            self.turns.append(new_turn)
            return self.parse_turns(id + 1)

    def create_conv(self):
        self.conv = {}
        self.conv.update({'domains': self.domains, 'dialogue_idx': self.dialog_idx})
        self.conv['dialogue'] = []
        i = 0
        for turn in self.turns:
            # print(turn.belief)
            self.conv['dialogue'].append({})
            self.conv['dialogue'][i].update({'turn_label': turn.turn_label,
                                             'domain': turn.domain,
                                             'system_transcript': turn.output,
                                             'turn_idx': turn.turn_idx,
                                             'belief_state': turn.belief,
                                             'transcript': turn.usr,
                                             'system_acts': turn.system_acts})
            i += 1

    def dictojson(self):

        json.dump(self.conv, open(f'{self.dialog_idx}.json', 'w'), sort_keys=False, indent=4)
        # print(transfo)

def listtojson(l,name):
    json.dump(l, open(f'{name}.json', 'w'), sort_keys=False, indent=4)


if __name__ == '__main__':
    PATH = "../dstc/data/"

    huge_list = []
    for dir in glob.glob(f'{PATH}*/**voip*'):
        first_conv = Converter(f"{dir}/label.json", f"{dir}/log.json")

        first_conv.getid()
        first_conv.parse_turns()
        first_conv.create_conv()
        huge_list.append(first_conv.conv)
        # print(first_conv.conv)
        # sys.exit()
    listtojson(huge_list,'../dstc/babiWoz')


import glob
import zipfile
import json
EVENT_DICTIONARY_V2 = {"Penalty":0,"Kick-off":1,"Goal":2,"Substitution":3,"Offside":4,"Shots on target":5,
                                "Shots off target":6,"Clearance":7,"Ball out of play":8,"Throw-in":9,"Foul":10,
                                "Indirect free-kick":11,"Direct free-kick":12,"Corner":13,"Yellow card":14
                                ,"Red card":15,"Yellow->red card":16}

INVERSE_EVENT_DICTIONARY_V2 = {0:"Penalty",1:"Kick-off",2:"Goal",3:"Substitution",4:"Offside",5:"Shots on target",
                                6:"Shots off target",7:"Clearance",8:"Ball out of play",9:"Throw-in",10:"Foul",
                                11:"Indirect free-kick",12:"Direct free-kick",13:"Corner",14:"Yellow card"
                                ,15:"Red card",16:"Yellow->red card"}

EVENT_DICTIONARY_V1 = {"card": 0, "subs": 1, "soccer": 2}

INVERSE_EVENT_DICTIONARY_V1 = {0: "card", 1: "subs", 2: "soccer"}


EVENT_DICTIONARY_BALL = {"PASS": 0, "DRIVE": 1}

INVERSE_EVENT_DICTIONARY_BALL = {0: "PASS", 1: "DRIVE"}

FRAME_CLASS_DICTIONARY = {"Ball":0,
						"Player team left":1,
						"Player team right":2,
						"Goalkeeper team left":3,
						"Goalkeeper team right":4,
						"Main referee":5,
						"Side referee":6,
						"Staff members":7,
						"Player team unknown 1":8,
						"Player team unknown 2":9,
						"Goalkeeper team unknown":10,
						"Goal left post left ":11,
						"Goal left post right":12,
						"Goal left crossbar":13,
						"Goal right post left":14,
						"Goal right post right":15,
						"Goal right crossbar":16,
						"Side line left":17,
						"Side line right":18,
						"Side line top":19,
						"Side line bottom":20,
						"Middle line":21,
						"Big rect. left top":22,
						"Big rect. left bottom":23,
						"Big rect. left main":24,
						"Big rect. right top":25,
						"Big rect. right bottom":26,
						"Big rect. right main":27,
						"Small rect. left top":28,
						"Small rect. left bottom":29,
						"Small rect. left main":30,
						"Small rect. right top":31,
						"Small rect. right bottom":32,
						"Small rect. right main":33,
						"Circle left":34,
						"Circle right":35,
						"Circle central":36,
						"Yellow card":37,
						"Red card":38,
						"Referee flag":39,
						"Wall of players":40,
						"Line unknown":41,
						"Goal unknown":42
}

INVERSE_FRAME_CLASS_DICTIONARY = {0:"Ball",
								1:"Player team left",
								2:"Player team right",
								3:"Goalkeeper team left",
								4:"Goalkeeper team right",
								5:"Main referee",
								6:"Side referee",
								7:"Staff members",
								8:"Player team unknown 1",
								9:"Player team unknown 2",
								10:"Goalkeeper team unknown",
								11:"Goal left post left ",
								12:"Goal left post right",
								13:"Goal left crossbar",
								14:"Goal right post left",
								15:"Goal right post right",
								16:"Goal right crossbar",
								17:"Side line left",
								18:"Side line right",
								19:"Side line top",
								20:"Side line bottom",
								21:"Middle line",
								22:"Big rect. left top",
								23:"Big rect. left bottom",
								24:"Big rect. left main",
								25:"Big rect. right top",
								26:"Big rect. right bottom",
								27:"Big rect. right main",
								28:"Small rect. left top",
								29:"Small rect. left bottom",
								30:"Small rect. left main",
								31:"Small rect. right top",
								32:"Small rect. right bottom",
								33:"Small rect. right main",
								34:"Circle left",
								35:"Circle right",
								36:"Circle central",
								37:"Yellow card",
								38:"Red card",
								39:"Referee flag",
								40:"Wall of players",
								41:"Line unknown",
								42:"Goal unknown"
}

FRAME_CLASS_COLOR_DICTIONARY = {'Ball': '#fb9b0b', 
								'Player team left': '#298af1', 
								'Player team right': '#ed2234', 
								'Goalkeeper team left': '#46d7f5', 
								'Goalkeeper team right': '#e03f91', 
								'Main referee': '#f9f611', 
								'Side referee': '#f3edb7', 
								'Staff members': '#0cf32d', 
								'Player team unknown 1': '#d9ecf6', 
								'Player team unknown 2': '#fadbe6', 
								'Goalkeeper team unknown': '#b7b7b7', 
								'Goal left post left ': '#2b84d3', 
								'Goal left post right': '#d92b2b', 
								'Goal left crossbar': '#16ec47', 
								'Goal right post left': '#f10bbc', 
								'Goal right post right': '#e0f012', 
								'Goal right crossbar': '#df8113', 
								'Side line left': '#15e8b2', 
								'Side line right': '#e60463', 
								'Side line top': '#e3bd17', 
								'Side line bottom': '#9a781e', 
								'Middle line': '#5d2bc3', 
								'Big rect. left top': '#d73b24', 
								'Big rect. left bottom': '#00a2b1', 
								'Big rect. left main': '#919cbf', 
								'Big rect. right top': '#c207e5', 
								'Big rect. right bottom': '#ae565f', 
								'Big rect. right main': '#0024ac', 
								'Small rect. left top': '#a94b94', 
								'Small rect. left bottom': '#cba060', 
								'Small rect. left main': '#c15cc1', 
								'Small rect. right top': '#f07b16', 
								'Small rect. right bottom': '#6427e3', 
								'Small rect. right main': '#ccc8e4', 
								'Circle left': '#0d4372', 
								'Circle right': '#b14f06', 
								'Circle central': '#2096c9', 
								'Yellow card': '#daf21b', 
								'Red card': '#fa150b', 
								'Referee flag': '#d5a263', 
								'Wall of players': '#ddd3b6', 
								'Line unknown': '#980dd2', 
								'Goal unknown': '#3e8bd4'
}

EVENT_DICTIONARY_CAPTION_V1 = {"corner" : 0,"substitution" : 0,"y-card" : 0,"whistle" : 0,"soccer-ball" : 0,"injury" : 0,"penalty" : 0,"yr-card" : 0,"r-card" : 0,"soccer-ball-own" : 0,"penalty-missed" : 0, "comments":0}
INVERSE_EVENT_DICTIONARY_CAPTION_V1 = {0 : "comments"}
EVENT_DICTIONARY_CAPTION_V2 = {"corner" : 0,"substitution" : 0,"y-card" : 0,"whistle" : 0,"soccer-ball" : 0,"injury" : 0,"penalty" : 0,"yr-card" : 0,"r-card" : 0,"soccer-ball-own" : 0,"penalty-missed" : 0, "" : 0, "comments":0}
INVERSE_EVENT_DICTIONARY_CAPTION_V2 = {0 : "comments"}


def getMetaDataTask(task, dataset, version):

	if dataset == "SoccerNet":
		if task == "caption":
			if version == 1:
				event_dict = EVENT_DICTIONARY_CAPTION_V1
				inv_dict = INVERSE_EVENT_DICTIONARY_CAPTION_V1
				num_classes = 1
				labels="Labels-caption.json"
			elif version == 2:
				event_dict = EVENT_DICTIONARY_CAPTION_V2
				inv_dict = INVERSE_EVENT_DICTIONARY_CAPTION_V2
				num_classes = 1
				labels="Labels-caption.json"
	return labels, num_classes, event_dict, inv_dict	

def LoadJsonFromZip(zippedFile, JsonPath):
    with zipfile.ZipFile(zippedFile, "r") as z:
        # print(filename)
        with z.open(JsonPath) as f:
            data = f.read()
            d = json.loads(data.decode("utf-8"))

    return d


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#!/usr/bin/env python3
'''
Agent Topology:
    Derived from Robinson & Goforth Topology.
    Flips column, to give layout:
                            COL:
                            C/0     D/1
            ROW:    C/0     R       S
                    D/1     T       P
                    
                            COL:
                            C/0     D/1
            ROW:    C/0     0,0     0,1
                    D/1     1,0     1,1
                    
'''
class agent_topology():

    def __init__(self):     
        
        self.gameforms = {
            "g111": [ [[3,3],[1,4]], [[4,1],[2,2]] ],
            "g112": [ [[3,3],[1,4]], [[4,2],[2,1]] ],
            "g113": [ [[3,2],[1,4]], [[4,3],[2,1]] ],
            "g114": [ [[3,1],[1,4]], [[4,3],[2,2]] ],
            "g115": [ [[3,1],[1,4]], [[4,2],[2,3]] ],
            "g116": [ [[3,2],[1,4]], [[4,1],[2,3]] ],
        
            "g161": [ [[2,3],[1,4]], [[4,1],[3,2]] ],
            "g162": [ [[2,3],[1,4]], [[4,2],[3,1]] ],
            "g163": [ [[2,2],[1,4]], [[4,3],[3,1]] ],
            "g164": [ [[2,1],[1,4]], [[4,3],[3,2]] ],
            "g165": [ [[2,1],[1,4]], [[4,2],[3,3]] ],
            "g166": [ [[2,2],[1,4]], [[4,1],[3,3]] ],
        
            "g151": [ [[1,3],[2,4]], [[4,1],[3,2]] ],
            "g152": [ [[1,3],[2,4]], [[4,2],[3,1]] ],
            "g153": [ [[1,2],[2,4]], [[4,3],[3,1]] ],
            "g154": [ [[1,1],[2,4]], [[4,3],[3,2]] ],
            "g155": [ [[1,1],[2,4]], [[4,2],[3,3]] ],
            "g156": [ [[1,2],[2,4]], [[4,1],[3,3]] ],
                                 
            "g141": [ [[1,3],[3,4]], [[4,1],[2,2]] ],
            "g142": [ [[1,3],[3,4]], [[4,2],[2,1]] ],
            "g143": [ [[1,2],[3,4]], [[4,3],[2,1]] ],
            "g144": [ [[1,1],[3,4]], [[4,3],[2,2]] ],
            "g145": [ [[1,1],[3,4]], [[4,2],[2,3]] ],
            "g146": [ [[1,2],[3,4]], [[4,1],[2,3]] ],
                                 
            "g131": [ [[2,3],[3,4]], [[4,1],[1,2]] ],
            "g132": [ [[2,3],[3,4]], [[4,2],[1,1]] ],
            "g133": [ [[2,2],[3,4]], [[4,3],[1,1]] ],
            "g134": [ [[2,1],[3,4]], [[4,3],[1,2]] ],
            "g135": [ [[2,1],[3,4]], [[4,2],[1,3]] ],
            "g136": [ [[2,2],[3,4]], [[4,1],[1,3]] ],
                                 
            "g121": [ [[3,3],[2,4]], [[4,1],[1,2]] ],
            "g122": [ [[3,3],[2,4]], [[4,2],[1,1]] ],
            "g123": [ [[3,2],[2,4]], [[4,3],[1,1]] ],
            "g124": [ [[3,1],[2,4]], [[4,3],[1,2]] ],
            "g125": [ [[3,1],[2,4]], [[4,2],[1,3]] ],
            "g126": [ [[3,2],[2,4]], [[4,1],[1,3]] ],
                                 
                                 
            "g211": [ [[4,3],[2,4]], [[3,1],[1,2]] ],
            "g212": [ [[4,3],[2,4]], [[3,2],[1,1]] ],
            "g213": [ [[4,2],[2,4]], [[3,3],[1,1]] ],
            "g214": [ [[4,1],[2,4]], [[3,3],[1,2]] ],
            "g215": [ [[4,1],[2,4]], [[3,2],[1,3]] ],
            "g216": [ [[4,2],[2,4]], [[3,1],[1,3]] ],
                                 
            "g261": [ [[4,3],[3,4]], [[2,1],[1,2]] ],
            "g262": [ [[4,3],[3,4]], [[2,2],[1,1]] ],
            "g263": [ [[4,2],[3,4]], [[2,3],[1,1]] ],
            "g264": [ [[4,1],[3,4]], [[2,3],[1,2]] ],
            "g265": [ [[4,1],[3,4]], [[2,2],[1,3]] ],
            "g266": [ [[4,2],[3,4]], [[2,1],[1,3]] ],
                                 
            "g251": [ [[4,3],[3,4]], [[1,1],[2,2]] ],
            "g252": [ [[4,3],[3,4]], [[1,2],[2,1]] ],
            "g253": [ [[4,2],[3,4]], [[1,3],[2,1]] ],
            "g254": [ [[4,1],[3,4]], [[1,3],[2,2]] ],
            "g255": [ [[4,1],[3,4]], [[1,2],[2,3]] ],
            "g256": [ [[4,2],[3,4]], [[1,1],[2,3]] ],
                                 
            "g241": [ [[4,3],[2,4]], [[1,1],[3,2]] ],
            "g242": [ [[4,3],[2,4]], [[1,2],[3,1]] ],
            "g243": [ [[4,2],[2,4]], [[1,3],[3,1]] ],
            "g244": [ [[4,1],[2,4]], [[1,3],[3,2]] ],
            "g245": [ [[4,1],[2,4]], [[1,2],[3,3]] ],
            "g246": [ [[4,2],[2,4]], [[1,1],[3,3]] ],
                                 
            "g231": [ [[4,3],[1,4]], [[2,1],[3,2]] ],
            "g232": [ [[4,3],[1,4]], [[2,2],[3,1]] ],
            "g233": [ [[4,2],[1,4]], [[2,3],[3,1]] ],
            "g234": [ [[4,1],[1,4]], [[2,3],[3,2]] ],
            "g235": [ [[4,1],[1,4]], [[2,2],[3,3]] ],
            "g236": [ [[4,2],[1,4]], [[2,1],[3,3]] ],
                                 
            "g221": [ [[4,3],[1,4]], [[3,1],[2,2]] ],
            "g222": [ [[4,3],[1,4]], [[3,2],[2,1]] ],
            "g223": [ [[4,2],[1,4]], [[3,3],[2,1]] ],
            "g224": [ [[4,1],[1,4]], [[3,3],[2,2]] ],
            "g225": [ [[4,1],[1,4]], [[3,2],[2,3]] ],
            "g226": [ [[4,2],[1,4]], [[3,1],[2,3]] ],
                                 
                                 
            "g311": [ [[4,4],[2,3]], [[3,2],[1,1]] ],
            "g312": [ [[4,4],[2,3]], [[3,1],[1,2]] ],
            "g313": [ [[4,4],[2,2]], [[3,1],[1,3]] ],
            "g314": [ [[4,4],[2,1]], [[3,2],[1,3]] ],
            "g315": [ [[4,4],[2,1]], [[3,3],[1,2]] ],
            "g316": [ [[4,4],[2,2]], [[3,3],[1,1]] ],
                                 
            "g361": [ [[4,4],[3,3]], [[2,2],[1,1]] ],
            "g362": [ [[4,4],[3,3]], [[2,1],[1,2]] ],
            "g363": [ [[4,4],[3,2]], [[2,1],[1,3]] ],
            "g364": [ [[4,4],[3,1]], [[2,2],[1,3]] ],
            "g365": [ [[4,4],[3,1]], [[2,3],[1,2]] ],
            "g366": [ [[4,4],[3,2]], [[2,3],[1,1]] ],
                                 
            "g351": [ [[4,4],[3,3]], [[1,2],[2,1]] ],
            "g352": [ [[4,4],[3,3]], [[1,1],[2,2]] ],
            "g353": [ [[4,4],[3,2]], [[1,1],[2,3]] ],
            "g354": [ [[4,4],[3,1]], [[1,2],[2,3]] ],
            "g355": [ [[4,4],[3,1]], [[1,3],[2,2]] ],
            "g356": [ [[4,4],[3,2]], [[1,3],[2,1]] ],
                                 
            "g341": [ [[4,4],[2,3]], [[1,2],[3,1]] ],
            "g342": [ [[4,4],[2,3]], [[1,1],[3,2]] ],
            "g343": [ [[4,4],[2,2]], [[1,1],[3,3]] ],
            "g344": [ [[4,4],[2,1]], [[1,2],[3,3]] ],
            "g345": [ [[4,4],[2,1]], [[1,3],[3,2]] ],
            "g346": [ [[4,4],[2,2]], [[1,3],[3,1]] ],
                                 
            "g331": [ [[4,4],[1,3]], [[2,2],[3,1]] ],
            "g332": [ [[4,4],[1,3]], [[2,1],[3,2]] ],
            "g333": [ [[4,4],[1,2]], [[2,1],[3,3]] ],
            "g334": [ [[4,4],[1,1]], [[2,2],[3,3]] ],
            "g335": [ [[4,4],[1,1]], [[2,3],[3,2]] ],
            "g336": [ [[4,4],[1,2]], [[2,3],[3,1]] ],
                                 
            "g321": [ [[4,4],[1,3]], [[3,2],[2,1]] ],
            "g322": [ [[4,4],[1,3]], [[3,1],[2,2]] ],
            "g323": [ [[4,4],[1,2]], [[3,1],[2,3]] ],
            "g324": [ [[4,4],[1,1]], [[3,2],[2,3]] ],
            "g325": [ [[4,4],[1,1]], [[3,3],[2,2]] ],
            "g326": [ [[4,4],[1,2]], [[3,3],[2,1]] ],
                                 
                                 
            "g411": [ [[3,4],[1,3]], [[4,2],[2,1]] ],
            "g412": [ [[3,4],[1,3]], [[4,1],[2,2]] ],
            "g413": [ [[3,4],[1,2]], [[4,1],[2,3]] ],
            "g414": [ [[3,4],[1,1]], [[4,2],[2,3]] ],
            "g415": [ [[3,4],[1,1]], [[4,3],[2,2]] ],
            "g416": [ [[3,4],[1,2]], [[4,3],[2,1]] ],
        
            "g461": [ [[2,4],[1,3]], [[4,2],[3,1]] ],
            "g462": [ [[2,4],[1,3]], [[4,1],[3,2]] ],
            "g463": [ [[2,4],[1,2]], [[4,1],[3,3]] ],
            "g464": [ [[2,4],[1,1]], [[4,2],[3,3]] ],
            "g465": [ [[2,4],[1,1]], [[4,3],[3,2]] ],
            "g466": [ [[2,4],[1,2]], [[4,3],[3,1]] ],
        
            "g451": [ [[1,4],[2,3]], [[4,2],[3,1]] ],
            "g452": [ [[1,4],[2,3]], [[4,1],[3,2]] ],
            "g453": [ [[1,4],[2,2]], [[4,1],[3,3]] ],
            "g454": [ [[1,4],[2,1]], [[4,2],[3,3]] ],
            "g455": [ [[1,4],[2,1]], [[4,3],[3,2]] ],
            "g456": [ [[1,4],[2,2]], [[4,3],[3,1]] ],
        
            "g441": [ [[1,4],[3,3]], [[4,2],[2,1]] ],
            "g442": [ [[1,4],[3,3]], [[4,1],[2,2]] ],
            "g443": [ [[1,4],[3,2]], [[4,1],[2,3]] ],
            "g444": [ [[1,4],[3,1]], [[4,2],[2,3]] ],
            "g445": [ [[1,4],[3,1]], [[4,3],[2,2]] ],
            "g446": [ [[1,4],[3,2]], [[4,3],[2,1]] ],
        
            "g431": [ [[2,4],[3,3]], [[4,2],[1,1]] ],
            "g432": [ [[2,4],[3,3]], [[4,1],[1,2]] ],
            "g433": [ [[2,4],[3,2]], [[4,1],[1,3]] ],
            "g434": [ [[2,4],[3,1]], [[4,2],[1,3]] ],
            "g435": [ [[2,4],[3,1]], [[4,3],[1,2]] ],
            "g436": [ [[2,4],[3,2]], [[4,3],[1,1]] ],
        
            "g421": [ [[3,4],[2,3]], [[4,2],[1,1]] ],
            "g422": [ [[3,4],[2,3]], [[4,1],[1,2]] ],
            "g423": [ [[3,4],[2,2]], [[4,1],[1,3]] ],
            "g424": [ [[3,4],[2,1]], [[4,2],[1,3]] ],
            "g425": [ [[3,4],[2,1]], [[4,3],[1,2]] ],
            "g426": [ [[3,4],[2,2]], [[4,3],[1,1]] ],
        }
        
        self.gameform_names = {

            "g111": "g111 - Prisoner\'s Dilemma",
            "g112": "g112",
            "g113": "g113",
            "g114": "g114",
            "g115": "g115",
            "g116": "g116",
            "g161": "g162",
            "g162": "g162",
            "g163": "g163",
            "g164": "g164",
            "g165": "g165",
            "g166": "g166",
            "g151": "g151",
            "g152": "g152",
            "g153": "g153",
            "g154": "g154",
            "g155": "g155",
            "g156": "g156",
            "g141": "g141",
            "g142": "g142",
            "g143": "g143",
            "g144": "g144",
            "g145": "g145",
            "g146": "g146",                   
            "g131": "g131",
            "g132": "g132",
            "g133": "g133",
            "g134": "g134",
            "g135": "g135",
            "g136": "g136",
            "g121": "g121",
            "g122": "g122 - Chicken",
            "g123": "g123",
            "g124": "g124",
            "g125": "g125",
            "g126": "g126",
            "g211": "g211",
            "g212": "g212",
            "g213": "g213",
            "g214": "g214",
            "g215": "g215",
            "g216": "g216",
            "g261": "g261",
            "g262": "g262",
            "g263": "g263",
            "g264": "g264",
            "g265": "g265",
            "g266": "g266",
            "g251": "g251",
            "g252": "g252",
            "g253": "g253",
            "g254": "g254",
            "g255": "g255",
            "g256": "g256",
            "g241": "g241",
            "g242": "g242",
            "g243": "g243",
            "g244": "g244",
            "g245": "g245",
            "g246": "g246",
            "g231": "g231",
            "g232": "g232",
            "g233": "g233",
            "g234": "g234",
            "g235": "g235",
            "g236": "g236",
            "g221": "g221",
            "g222": "g222",
            "g223": "g223",
            "g224": "g224",
            "g225": "g225",
            "g226": "g226",
            "g311": "g311",
            "g312": "g312",
            "g313": "g313",
            "g314": "g314",
            "g315": "g315",
            "g316": "g316",
            "g361": "g361",
            "g362": "g362",
            "g363": "g363",
            "g364": "g364",
            "g365": "g365",
            "g366": "g366",
            "g351": "g351",
            "g352": "g352",
            "g353": "g353",
            "g354": "g354",
            "g355": "g355",
            "g356": "g356",
            "g341": "g341",
            "g342": "g342",
            "g343": "g343",
            "g344": "g344",
            "g345": "g345",
            "g346": "g346",
            "g331": "g331",
            "g332": "g332",
            "g333": "g333",
            "g334": "g334",
            "g335": "g335",
            "g336": "g336",
            "g321": "g321",
            "g322": "g322 - Stag Hunt",
            "g323": "g323",
            "g324": "g324",
            "g325": "g325",
            "g326": "g326",
            "g411": "g411",
            "g412": "g412",
            "g413": "g413",
            "g414": "g414",
            "g415": "g415",
            "g416": "g416",
            "g461": "g461",
            "g462": "g462",
            "g463": "g463",
            "g464": "g464",
            "g465": "g465",
            "g466": "g466",
            "g451": "g451",
            "g452": "g452",
            "g453": "g453",
            "g454": "g454",
            "g455": "g455",
            "g456": "g456",
            "g441": "g441",
            "g442": "g442",
            "g443": "g443",
            "g444": "g444",
            "g445": "g445",
            "g446": "g446",
            "g431": "g431",
            "g432": "g432",
            "g433": "g433",
            "g434": "g434",
            "g435": "g435",
            "g436": "g436",
            "g421": "g421",
            "g422": "g422",
            "g423": "g423",
            "g424": "g424",
            "g425": "g425",
            "g426": "g426",
        }
        
    def heatmap_model(self):
    
        data = [
        [[{"g212": 0}], [{"g213": 0}], [{"g214": 0}], [{"g215": 0}], [{"g216": 0}], [{"g211": 0}],      [{"g312": 0}], [{"g313": 0}], [{"g314": 0}], [{"g315": 0}], [{"g316": 0}], [{"g311": 0}],],
        [[{"g262": 0}], [{"g263": 0}], [{"g264": 0}], [{"g265": 0}], [{"g266": 0}], [{"g261": 0}],      [{"g362": 0}], [{"g363": 0}], [{"g364": 0}], [{"g365": 0}], [{"g366": 0}], [{"g361": 0}],],
        [[{"g252": 0}], [{"g253": 0}], [{"g254": 0}], [{"g255": 0}], [{"g256": 0}], [{"g251": 0}],      [{"g352": 0}], [{"g353": 0}], [{"g354": 0}], [{"g355": 0}], [{"g356": 0}], [{"g351": 0}],],
        [[{"g242": 0}], [{"g243": 0}], [{"g244": 0}], [{"g245": 0}], [{"g246": 0}], [{"g241": 0}],      [{"g342": 0}], [{"g343": 0}], [{"g344": 0}], [{"g345": 0}], [{"g346": 0}], [{"g341": 0}],],
        [[{"g232": 0}], [{"g233": 0}], [{"g234": 0}], [{"g235": 0}], [{"g236": 0}], [{"g231": 0}],      [{"g332": 0}], [{"g333": 0}], [{"g334": 0}], [{"g335": 0}], [{"g336": 0}], [{"g331": 0}],],
        [[{"g222": 0}], [{"g223": 0}], [{"g224": 0}], [{"g225": 0}], [{"g226": 0}], [{"g221": 0}],      [{"g322": 0}], [{"g323": 0}], [{"g324": 0}], [{"g325": 0}], [{"g326": 0}], [{"g321": 0}],],
        
        [[{"g112": 0}], [{"g113": 0}], [{"g114": 0}], [{"g115": 0}], [{"g116": 0}], [{"g111": 0}],      [{"g412": 0}], [{"g313": 0}], [{"g314": 0}], [{"g315": 0}], [{"g316": 0}], [{"g311": 0}],],
        [[{"g162": 0}], [{"g163": 0}], [{"g164": 0}], [{"g165": 0}], [{"g166": 0}], [{"g161": 0}],      [{"g462": 0}], [{"g463": 0}], [{"g464": 0}], [{"g465": 0}], [{"g466": 0}], [{"g461": 0}],],
        [[{"g152": 0}], [{"g153": 0}], [{"g154": 0}], [{"g155": 0}], [{"g156": 0}], [{"g151": 0}],      [{"g452": 0}], [{"g453": 0}], [{"g454": 0}], [{"g455": 0}], [{"g456": 0}], [{"g451": 0}],],
        [[{"g142": 0}], [{"g143": 0}], [{"g144": 0}], [{"g145": 0}], [{"g146": 0}], [{"g141": 0}],      [{"g442": 0}], [{"g443": 0}], [{"g444": 0}], [{"g445": 0}], [{"g446": 0}], [{"g441": 0}],],
        [[{"g132": 0}], [{"g133": 0}], [{"g134": 0}], [{"g135": 0}], [{"g136": 0}], [{"g131": 0}],      [{"g432": 0}], [{"g433": 0}], [{"g434": 0}], [{"g435": 0}], [{"g436": 0}], [{"g431": 0}],],
        [[{"g122": 0}], [{"g123": 0}], [{"g124": 0}], [{"g125": 0}], [{"g126": 0}], [{"g121": 0}],      [{"g422": 0}], [{"g423": 0}], [{"g424": 0}], [{"g425": 0}], [{"g426": 0}], [{"g421": 0}],],
        ]
        
        return data
    
    def nbs_location_table(self):
        
        data = {
            "g111": [[0,0],[None]],
            "g112": [[1,0],[None]],
            "g113": [[1,0],[0,1]],
            "g114": [[1,0],[0,1]],
            "g115": [[1,0],[0,1]],
            "g116": [[1,1],[None]],
            "g121": [[0,1],[None]],
            "g122": [[0,0],[None]],
            "g123": [[1,0],[0,1]],
            "g124": [[1,0],[0,1]],
            "g125": [[1,0],[0,1]],
            "g126": [[1,0],[0,1]],
            "g131": [[1,1],[0,1]],
            "g132": [[1,0],[0,1]],
            "g133": [[1,0],[0,1]],
            "g134": [[1,0],[0,1]],
            "g135": [[0,1],[None]],
            "g136": [[0,1],[None]],
            "g141": [[1,0],[0,1]],
            "g142": [[1,0],[0,1]],
            "g143": [[1,0],[0,1]],
            "g144": [[1,0],[0,1]],
            "g145": [[0,1],[None]],
            "g146": [[0,1],[None]],
            "g151": [[1,0],[0,1]],
            "g152": [[1,0],[0,1]],
            "g153": [[1,0],[None]],
            "g154": [[1,0],[None]],
            "g155": [[1,1],[None]],
            "g156": [[1,1],[None]],
            "g161": [[1,1],[None]],
            "g162": [[1,0],[0,1]],
            "g163": [[1,0],[None]],
            "g164": [[1,0],[None]],
            "g165": [[1,1],[None]],
            "g166": [[1,1],[None]],
            "g211": [[0,0],[None]],
            "g212": [[0,0],[None]],
            "g213": [[1,0],[None]],
            "g214": [[1,0],[None]],
            "g215": [[0,0],[0,1]],
            "g216": [[0,0],[0,1]],
            "g221": [[0,0],[None]],
            "g222": [[0,0],[None]],
            "g223": [[1,0],[None]],
            "g224": [[1,0],[None]],
            "g225": [[1,1],[None]],
            "g226": [[0,0],[0,1]],
            "g231": [[0,0],[None]],
            "g232": [[0,0],[None]],
            "g233": [[0,0],[0,1]],
            "g234": [[1,1],[None]],
            "g235": [[1,1],[0,1]],
            "g236": [[1,1],[0,1]],
            "g241": [[0,0],[1,1]],
            "g242": [[0,0],[None]],
            "g243": [[0,0],[0,1]],
            "g244": [[0,0],[0,1]],
            "g245": [[1,1],[0,1]],
            "g246": [[1,1],[0,1]],
            "g251": [[0,0],[None]],
            "g252": [[0,0],[None]],
            "g253": [[0,0],[0,1]],
            "g254": [[0,0],[0,1]],
            "g255": [[0,0],[0,1]],
            "g256": [[0,0],[0,1]],
            "g261": [[0,0],[None]],
            "g262": [[0,0],[None]],
            "g263": [[0,0],[0,1]],
            "g264": [[0,0],[0,1]],
            "g265": [[0,0],[0,1]],
            "g266": [[0,0],[0,1]],
            "g311": [[0,0],[None]],
            "g312": [[0,0],[None]],
            "g313": [[0,0],[None]],
            "g314": [[0,0],[None]],
            "g315": [[0,0],[None]],
            "g316": [[0,0],[None]],
            "g321": [[0,0],[None]],
            "g322": [[0,0],[None]],
            "g323": [[0,0],[None]],
            "g324": [[0,0],[None]],
            "g325": [[0,0],[None]],
            "g326": [[0,0],[None]],
            "g331": [[0,0],[None]],
            "g332": [[0,0],[None]],
            "g333": [[0,0],[None]],
            "g334": [[0,0],[None]],
            "g335": [[0,0],[None]],
            "g336": [[0,0],[None]],
            "g341": [[0,0],[None]],
            "g342": [[0,0],[None]],
            "g343": [[0,0],[None]],
            "g344": [[0,0],[None]],
            "g345": [[0,0],[None]],
            "g346": [[0,0],[None]],
            "g351": [[0,0],[None]],
            "g352": [[0,0],[None]],
            "g353": [[0,0],[None]],
            "g354": [[0,0],[None]],
            "g355": [[0,0],[None]],
            "g356": [[0,0],[None]],
            "g361": [[0,0],[None]],
            "g362": [[0,0],[None]],
            "g363": [[0,0],[None]],
            "g364": [[0,0],[None]],
            "g365": [[0,0],[None]],
            "g366": [[0,0],[None]],
            "g411": [[0,0],[None]],
            "g412": [[0,0],[None]],
            "g413": [[0,0],[None]],
            "g414": [[0,0],[None]],
            "g415": [[0,0],[None]],
            "g416": [[0,0],[None]],
            "g421": [[0,0],[None]],
            "g422": [[0,0],[None]],
            "g423": [[0,0],[None]],
            "g424": [[0,0],[None]],
            "g425": [[0,0],[None]],
            "g426": [[0,0],[None]],
            "g431": [[1,0],[0,1]],
            "g432": [[0,1],[None]],
            "g433": [[1,0],[0,0]],
            "g434": [[1,0],[0,0]],
            "g435": [[1,0],[0,0]],
            "g436": [[1,0],[0,0]],
            "g441": [[1,0],[0,1]],
            "g442": [[0,0],[None]],
            "g443": [[1,1],[None]],
            "g444": [[1,0],[0,0]],
            "g445": [[1,0],[0,0]],
            "g446": [[1,0],[0,0]],
            "g451": [[1,0],[0,0]],
            "g452": [[1,1],[None]],
            "g453": [[1,0],[1,1]],
            "g454": [[1,0],[1,1]],
            "g455": [[1,0],[0,0]],
            "g456": [[1,0],[0,0]],
            "g461": [[1,0],[0,0]],
            "g462": [[1,0],[0,0]],
            "g463": [[1,0],[1,1]],
            "g464": [[1,0],[1,1]],
            "g465": [[1,0],[0,0]],
            "g466": [[1,0],[0,0]]
            
        }
        
        return data
        
        
        
        
        
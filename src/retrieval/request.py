import requests

# URL for your local FastAPI server
url = "http://[fdbd:dccd:cdc2:12c8:0:28d::]:10446/retrieve"
# url = "http://127.0.0.1:8000/retrieve"
# url = "http://[::]:8000/retrieve"

test_rewrite = """Did Harry Hopkins work with other people to publicize and defend New Deal relief programs? Hopkins began important roles in New Deal programs such as Federal Emergency Relief Administration (FERA) in 1933. This period saw Hopkins as federal relief administrator. As part of his work, Hopkins supervised programs like FERA and Civil Works Administration (CWA). In his position, Hopkins was involved from 1933 to 1935 with these efforts. After joining Roosevelt's administration, Hopkins was extensively involved in relief projects. By the late 1930s, as he was fighting stomach cancer, FDR began to train Hopkins as a possible successor. Hopkins also supervised the Works Progress Administration (WPA). Throughout Hopkins' career, was he involved with publicizing and defending New Deal programs? In an effort to publicize and defend these programs, did Hopkins work with any particular groups or individuals? Some reports suggest Hopkins himself was a key public figure in promoting New Deal programs and sometimes worked with respected experts. Historically, there is evidence that Hopkins worked with and supported other government and public figures in advocating for FDR's New Deal programs. Additionally, came Hopkins' work with publicizing these programs include working with others in his capacity as federal relief administrator or in running various programs like FERA, FDR training him as potential successor, and his involvement with relief programs during the Great Depression. Did Hopkins collaborate with others to promote and protect New Deal programs, and when did these efforts start? Did Hopkins work with individuals like Edwin Campell, who helped to develop the New Deal and attempted to publicize government programs to the public as part of this role? How else did Hopkins work with these programs to ensure public support?"""
test_qid = "QReCC-Train_938_7"
# Example payload
payload = {
    # for topiocqa
    # "queries": ["what was australia's contribution to the battle of normandy?", "was the battle fought in australia?"],
    # "query_ids": ['topiocqa-train_1_1', 'topiocqa-train_1_2'], # gold id: 5498209, 5498207

    # for qrecc
    "queries": [test_rewrite, "How did Van Halen reunite with Roth?", "Where was he born?", 'Who is the new chairman of National Scheduled Tribes Commision', 'For how long was the country devided before it became united'],
    "query_ids": [test_qid, 'QReCC-Train_2_1', 'QReCC-Train_5_1', 'QReCC-Train_10822_1', 'QReCC-Train_10810_3'], # gold id: 54537936, 54572406, 54302883
    
    "topk": 100,
    "return_scores": False
}

# Send POST request
response = requests.post(url, json=payload)

# Raise an exception if the request failed
response.raise_for_status()

# Get the JSON response
retrieved_data = response.json()

print("Response from server:")
print(retrieved_data)

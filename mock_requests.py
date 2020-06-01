import requests
host = "http://localhost:8891"

# train
r = requests.get(f"{host}/train")
print(r.text)

# items_to_user
payload_1 = {"user_id": "12947", "nrec_items": "10", "show_known": "True"}
r = requests.post(f"{host}/items_to_user", json=payload_1)
print(r.text)

# users_to_item
payload_2 = {"item_id": "22386", "len_users": "15"}
r = requests.post(f"{host}/users_to_item", json=payload_2)
print(r.text)

# items_to_item
payload_3 = {"item_id": "22466", "n_items": "11"}
r = requests.post(f"{host}/items_to_item", json=payload_3)
print(r.text)

# wipe
r = requests.get(f"{host}/wipe")
print(r.text)


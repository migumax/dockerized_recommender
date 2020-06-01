# items_to_user
curl -X POST "http://127.0.0.1:8891/items_to_user" -H "Content-Type: application/json" -d '{"user_id": "12947", "nrec_items": "10", "show_known": "True"}'

# users_to_item
curl -X POST "http://127.0.0.1:8891/users_to_item" -H "Content-Type: application/json" -d '{"item_id": "22386", "len_users": "15"}'

# items_to_item
curl -X POST "http://127.0.0.1:8891/items_to_item" -H "Content-Type: application/json" -d '{"item_id": "22466", "n_items": "11"}'

# wipe
curl -X GET "http://127.0.0.1:8891/wipe" 

# train
curl -X GET "http://127.0.0.1:8891/train" 
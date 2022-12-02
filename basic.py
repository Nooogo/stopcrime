import pickle

# pickle로 저장하기 
def save_pickle(file_name, data): # file_name = file_name.pkl
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# pickle로 불러오기
def load_pickle(file_name): # file_name = file_name.pkl
    # 데이터 로드
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data 
import pandas as pd
movies = pd.read_csv('/content/drive/MyDrive/movies.csv')
rating = pd.read_csv('/content/drive/MyDrive/ratings_small.csv')
movies.head(10)
movies_pop = movies.drop(['genres', 'popularity', 'runtime', 'vote_average'], axis=1, errors='ignore')
userid_200 = rating['userId'].unique()[:200]
ratingby_200 = rating[rating['userId'].isin(userid_200)]
import numpy as np
movieIds = np.sort(ratingby_200['movieId'].unique())
user_to_index = {int(user_id): idx for idx, user_id in enumerate(userid_200)}
movie_to_index = {int(movie_id): idx for idx, movie_id in enumerate(movieIds)}
rows = ratingby_200['userId'].map(user_to_index)
cols = ratingby_200['movieId'].map(movie_to_index)
data = ratingby_200['rating'].astype(float)
rows = pd.Series(rows)
cols = pd.Series(cols)
data = pd.Series(data)
print(rows)
print(cols)
print(data)
from scipy.sparse import csr_matrix
valid_mask = rows.notna() & cols.notna()
rows = rows[valid_mask].astype(int)
cols = cols[valid_mask].astype(int)
data = data[valid_mask]
user_movie_matrix = csr_matrix((data, (rows, cols)), shape=(len(userid_200), len(movieIds)))
print(user_movie_matrix)
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix)
query_index = np.random.choice(user_movie_matrix.shape[0])  #Chọn ngẫu nhiên một người dùng (theo chỉ số dòng trong ma trận).
query_vector = user_movie_matrix[query_index] # Lấy vector đặc trưng (sở thích xem phim) của người dùng đó.
distances, indices = model_knn.kneighbors(query_vector, n_neighbors=6) #distances: chứa khoảng cách cosine giữa người này và 6 người giống nhất, indices: chứa vị trí (chỉ số hàng) của 6 người tương tự đó trong ma trận.
query_user_id = userid_200[query_index] # Lấy ID người dùng mà ta đang muốn gợi ý phim.
print(f" Recommendations for userId {query_user_id}:\n")
seen_movies = ratingby_200[ratingby_200['userId'] == query_user_id]['movieId'].values #Lấy danh sách các phim mà người dùng đã xem.
for i in range(1, len(distances.flatten())):
    similar_user_idx = indices.flatten()[i]
    similar_user_id = userid_200[similar_user_idx] #Lấy ID người dùng tương tự.
    similar_ratings = ratingby_200[ratingby_200['userId'] == similar_user_id] # Lọc ra các phim mà người dùng tương tự đã xem nhưng người dùng chính chưa xem.
    unseen_movies = similar_ratings[~similar_ratings['movieId'].isin(seen_movies)]
    if not unseen_movies.empty: #Nếu có phim chưa xem:
        top_movie = unseen_movies.sort_values(by='rating', ascending=False).iloc[0] #Lấy phim được đánh giá cao nhất trong số đó.
        movie_id = top_movie['movieId']
        title = movies_pop[movies_pop['movieId'] == movie_id]['title'].values # Lấy tên phim từ bảng thông tin phim.
        print(f"{i}: {title[0]}") #In ra gợi ý.
    else:
        print(f"{i}: Không có phim phù hợp để gợi ý từ userId {similar_user_id}")

        #Chọn người dùng cần gợi ý. Tìm các người dùng tương tự. Gợi ý phim mà họ thích nhưng người dùng chính chưa xem.
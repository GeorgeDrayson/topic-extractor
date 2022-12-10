import os
import markdown
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import argparse

def get_posts(posts_folder):

    posts_filepath = posts_folder
    list_of_posts = os.listdir(posts_filepath)
    posts = []

    for post in list_of_posts:
        post_filepath = posts_filepath + '/' + post
        with open(post_filepath, 'r') as f:
            text = f.read()
            content = markdown.markdown(text)
            text = content.split("</p>\n<hr />\n<p>")[1]
            text = text.replace('li','')
            posts.append(text)
    return(posts)


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_keywords(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract topics from markdown files')
    parser.add_argument("posts_folder", type=str, help='Filepath of posts')

    args = parser.parse_args()

    #Get posts
    posts = get_posts(args.posts_folder)

    #Initialise count vectorizer and transformer
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 5), max_features=10000)
    transformer = TfidfTransformer(smooth_idf=True,use_idf=True)

    #Returns document term matrix and fit transformer to it
    word_count_vector = vectorizer.fit_transform(posts)
    transformer.fit(word_count_vector)

    #Get tf-idf vector of particular post
    for i in range(len(posts)):
        tf_idf_vector = transformer.transform(vectorizer.transform([posts[i]]))
        sorted_items=sort_coo(tf_idf_vector.tocoo())

        #Get feature names and sort key words
        feature_names=vectorizer.get_feature_names_out()
        keywords=extract_keywords(feature_names,sorted_items,15)

        print("\n===Post " + str(i+1) + "===")
        for k in keywords:
            print(k,keywords[k])

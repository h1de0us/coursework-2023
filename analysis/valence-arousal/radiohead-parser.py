import requests
import base64

# NO MORE SPOTIPY

def parse_spotify(client_id, client_secret, filename):
    # Define endpoint URLs
    auth_url = 'https://accounts.spotify.com/api/token'
    search_url = 'https://api.spotify.com/v1/search'
    audio_features_url = 'https://api.spotify.com/v1/audio-features'

    # Request an access token from Spotify
    auth_header = base64.b64encode((client_id + ':' + client_secret).encode('ascii')).decode('ascii')
    auth_payload = {'grant_type': 'client_credentials'}
    auth_response = requests.post(auth_url, headers={'Authorization': 'Basic ' + auth_header}, data=auth_payload)
    access_token = auth_response.json()['access_token']

    # Define search query and parameters
    query = 'artist:"Radiohead"'
    query_params = {'q': query, 'type': 'track', 'market': 'US', 'limit': 50}

    # Send search request and get list of Radiohead tracks
    tracks = []
    while True:
        search_response = requests.get(search_url, params=query_params, headers={'Authorization': 'Bearer ' + access_token})
        tracks_response = search_response.json()['tracks']
        tracks.extend(tracks_response['items'])
        if not tracks_response['next']:
            break
        query_params['offset'] = tracks_response['offset'] + tracks_response['limit']

    # Get the audio features for each track
    audio_features = []
    for track in tracks:
        track_id = track['id']
        if track['artists'][0]['name'] == 'Radiohead':
            features_response = requests.get(audio_features_url + '/' + track_id, headers={'Authorization': 'Bearer ' + access_token})
            features = features_response.json()
            audio_features.append((track['name'], features))

    # print the audio features for each Radiohead song

    with open(filename) as file:
        for query in audio_features:
            file.write(f"{query[0]}: arousal={query[1]['energy']}, valence={query[1]['valence']}\n")

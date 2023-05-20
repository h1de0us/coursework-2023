import requests
import base64
import json
import os

def get_access_token():
    url = "https://accounts.spotify.com/api/token"
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')

    client_creds = f"{client_id}:{client_secret}"
    client_creds_b64 = base64.b64encode(client_creds.encode())

    headers = {
        "Authorization": f"Basic {client_creds_b64.decode()}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # Set up the request body
    data = {
        "grant_type": "client_credentials"
    }

    # Make the POST request to the API endpoint
    response = requests.post(url, headers=headers, data=data)

    # Parse the response and get the access token
    access_token = None
    if response.status_code == 200:
        response_data = response.json()
        access_token = response_data["access_token"]
        # print(f"Access token: {access_token}")
        return access_token
    else:
        print("Failed to get access token")
        return None


def load_data_from_spotify(access_token: str, tracks: dict):
    for track, id in tracks.items():
        track_id = id
        url = f"https://api.spotify.com/v1/tracks/{track_id}"
        headers = {
            "Authorization": f"Bearer {access_token}"
        }

        response = requests.get(url, headers=headers)

        data = json.loads(response.text)
        for key, value in data.items():
            print(key, value)


def get_track_ids(access_token, track_names):
    endpoint = "https://api.spotify.com/v1/search"
    tracks = {}
    for track_name in track_names:
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        params = {
            "q": f"track:{track_name}",
            "type": "track",
            "limit": 1
        }

        response = requests.get(endpoint, headers=headers, params=params)

        if response.status_code == 200:
            response_data = json.loads(response.text)
            track_id = response_data["tracks"]["items"][0]["id"]
            print(f"Track name: {track_name}, Track ID: {track_id}")
            tracks[track_name] = track_id
        else:
            print("Failed to get track ID")

    return tracks


# filename is a path to match_score.json
def match_songs_with_hdf5(filename, tracks):
    with open(filename) as f:
        data = json.load(f)
        for million_songs_key, v in data.items():
            for idd, val in v.items():
                print(idd)
                for track, spotify_id in tracks.items():
                    if idd == spotify_id:
                        print(track, million_songs_key, spotify_id)



def get_valence_and_arousal(access_token, track_names):
    for track_name in track_names:
        url = f'https://api.spotify.com/v1/search?q={track_name}&type=track&market=US&limit=1'
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

        response = requests.get(url, headers=headers)
        data = json.loads(response.text)
        track_id = data['tracks']['items'][0]['id']

        url = f'https://api.spotify.com/v1/audio-features/{track_id}'
        response = requests.get(url, headers=headers)
        data = json.loads(response.text)

        # extract the valence and arousal values and scale them from -1 to 1
        valence = 2 * data['valence'] - 1
        arousal = 2 * data['energy'] - 1

        print(f"Track: {track_name}, Valence: {valence}, Arousal: {arousal}")


if __name__ == '__main__':
    access_token = get_access_token()
    track_names = [
        # 'dancing queen, ABBA',
        'fake plastic trees, radiohead',
        # 'life on mars, david bowie',
        'high and dry, radiohead',
        'karma police, radiohead',
        'airbag, radiohead',
        'paranoid android, radiohead',
        '15 step, radiohead',
        'all i need, radiohead',
        'videotape, radiohead',
        'everything in its right place, radiohead',
        'true love waits, radiohead',
        'daydreaming, radiohead',
        'anyone can play guitar, radiohead'
        'no surprises, radiohead',
        'separator, radiohead',
        'pyramid song, radiohead',
        'dancing queen, abba',
        'slipping through my fingers, abba'

    ]
    # tracks = get_track_ids(access_token, track_names)
    # match_songs_with_hdf5('/Users/h1de0us/uni/coursework/project/match_scores.json', tracks)

    get_valence_and_arousal(access_token, track_names)



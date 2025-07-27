import random
import time
from typing import Dict, List, Tuple

import pandas as pd
import requests


def get_track_data_dataframe(
    track_ids: List[str], delay_between_batches: float = 0.1, verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Fetch track id from ReccoBeats API

    Args:
        track_ids: List of track IDs to query (any length)
        delay_between_batches: Seconds to wait between API calls (default 0.1)
        verbose: Whether to print progress information

    Returns:
        Tuple of (dataframe, stats_dict)
        - dataframe: DataFrame with all found tracks and their data
        - stats_dict: Dictionary with processing statistics
    """
    base_url = "https://api.reccobeats.com/v1/track"
    headers = {"Accept": "application/json"}
    batch_size = 40

    # Remove duplicates while preserving order
    unique_track_ids = list(dict.fromkeys(track_ids))

    all_tracks = []

    stats = {
        "input_total": len(track_ids),
        "input_unique": len(unique_track_ids),
        "duplicates_removed": len(track_ids) - len(unique_track_ids),
        "batches_total": (len(unique_track_ids) + batch_size - 1) // batch_size,
        "batches_successful": 0,
        "batches_failed": 0,
        "tracks_found": 0,
        "tracks_missing": 0,
        "failed_batches": [],
    }

    if verbose:
        print(
            f"Processing {stats['input_total']} IDs ({stats['input_unique']} unique) in {stats['batches_total']} batches..."
        )

    # Process IDs in batches of 40
    for batch_num, i in enumerate(range(0, len(unique_track_ids), batch_size), 1):
        batch = unique_track_ids[i : i + batch_size]

        if verbose and batch_num % 10 == 0:
            print(f"Processing batch {batch_num}/{stats['batches_total']}...")

        try:
            # Make API request with comma-separated IDs
            params = {"ids": ",".join(batch)}
            response = requests.get(
                base_url, headers=headers, params=params, timeout=30
            )
            response.raise_for_status()

            # Parse JSON response
            data = response.json()

            if isinstance(data, dict) and "content" in data:
                stats["batches_successful"] += 1
                batch_tracks = data["content"]

                # Add tracks to our collection
                for track in batch_tracks:
                    all_tracks.append(
                        {
                            "api_id": track.get("id"),
                            "track_title": track.get("trackTitle"),
                            "artists": track.get("artists"),
                            "duration_ms": track.get("durationMs"),
                            "isrc": track.get("isrc"),
                            "ean": track.get("ean"),
                            "upc": track.get("upc"),
                            "href": track.get("href"),
                            "available_countries": track.get("availableCountries"),
                            "popularity": track.get("popularity"),
                        }
                    )

                stats["tracks_found"] += len(batch_tracks)
                stats["tracks_missing"] += len(batch) - len(batch_tracks)

                if verbose and len(batch_tracks) != len(batch):
                    print(
                        f"Batch {batch_num}: {len(batch_tracks)}/{len(batch)} tracks found"
                    )

            else:
                print(f"Unexpected response format for batch {batch_num}: {type(data)}")
                if verbose:
                    print(
                        f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}"
                    )
                stats["batches_failed"] += 1
                stats["failed_batches"].append(batch_num)
                stats["tracks_missing"] += len(batch)

        except Exception as e:
            print(f"Error in batch {batch_num}: {e}")
            stats["batches_failed"] += 1
            stats["failed_batches"].append(batch_num)
            stats["tracks_missing"] += len(batch)

        # Add delay between requests to avoid rate limiting
        if batch_num < stats["batches_total"]:
            time.sleep(delay_between_batches)

    # Create DataFrame
    df = pd.DataFrame(all_tracks)

    if verbose:
        print("\nProcessing complete!")
        print(f"Input: {stats['input_total']} total, {stats['input_unique']} unique")
        print(
            f"Found: {stats['tracks_found']} tracks, Missing: {stats['tracks_missing']} tracks"
        )
        print(
            f"Batches: {stats['batches_successful']} successful, {stats['batches_failed']} failed"
        )
        print(f"DataFrame shape: {df.shape}")

        if stats["failed_batches"]:
            print(f"Failed batches: {stats['failed_batches']}")

    return df, stats


def get_track_ids_only(
    track_ids: List[str], **kwargs
) -> Tuple[List[str], Dict[str, int]]:
    """
    Simplified version that returns just the API IDs that were found.
    Note: Cannot maintain 1:1 mapping with input since API returns different IDs.

    Args:
        track_ids: List of track IDs to query
        **kwargs: Additional arguments passed to get_track_data_dataframe

    Returns:
        Tuple of (api_ids_list, stats_dict)
        - api_ids_list: List of API IDs that were found (shorter than input if some missing)
        - stats_dict: Dictionary with processing statistics
    """
    df, stats = get_track_data_dataframe(track_ids, **kwargs)
    api_ids = df["api_id"].dropna().tolist()
    return api_ids, stats


def get_track_audio_features(
    track_ids: List[str],
    delay_between_requests: float = 0.5,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch audio features for tracks from ReccoBeats API with rate limiting.

    Args:
        track_ids: List of track IDs to query
        delay_between_requests: Seconds to wait between requests (default 0.5)
        max_retries: Maximum number of retries for failed requests
        backoff_factor: Multiplier for delay on retries (exponential backoff)
        verbose: Whether to print progress information

    Returns:
        DataFrame with audio features for all successfully fetched tracks
    """
    base_url = "https://api.reccobeats.com/v1/track"
    headers = {"Accept": "application/json"}

    successful_responses = []
    failed_tracks = []

    if verbose:
        print(f"Fetching audio features for {len(track_ids)} tracks...")

    for i, track_id in enumerate(track_ids):
        if verbose and (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{len(track_ids)} tracks processed...")

        url = f"{base_url}/{track_id}/audio-features"

        # Retry logic with exponential backoff
        success = False
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code == 429:  # Rate limited
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = float(retry_after)
                        if verbose:
                            print(f"Rate limited. Waiting {wait_time} seconds...")
                    else:
                        # If no Retry-After header, use exponential backoff
                        wait_time = delay_between_requests * (
                            backoff_factor**attempt
                        ) + random.uniform(0, 1)
                        if verbose:
                            print(
                                f"Rate limited (attempt {attempt + 1}). Waiting {wait_time:.1f} seconds..."
                            )

                    time.sleep(wait_time)
                    continue

                response.raise_for_status()  # Raises exception for other HTTP errors
                data = response.json()

                # Add track_id to the response data for reference
                data["track_id"] = track_id
                successful_responses.append(data)
                success = True
                break

            except requests.exceptions.Timeout:
                if verbose:
                    print(f"Timeout for track {track_id} (attempt {attempt + 1})")
                if attempt < max_retries:
                    time.sleep(delay_between_requests * (backoff_factor**attempt))

            except requests.exceptions.RequestException as e:
                if verbose:
                    print(f"Error fetching {track_id} (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    time.sleep(delay_between_requests * (backoff_factor**attempt))

        if not success:
            failed_tracks.append(track_id)
            if verbose:
                print(f"Failed to fetch audio features for track: {track_id}")

        # Add delay between requests to avoid rate limiting
        if i < len(track_ids) - 1:  # Don't wait after the last request
            time.sleep(delay_between_requests)

    if verbose:
        print("\nCompleted!")
        print(f"Successful: {len(successful_responses)}")
        print(f"Failed: {len(failed_tracks)}")
        if failed_tracks:
            print(
                f"Failed track IDs: {failed_tracks[:5]}{'...' if len(failed_tracks) > 5 else ''}"
            )

    # Convert to DataFrame
    if successful_responses:
        df = pd.DataFrame(successful_responses)
        return df
    else:
        # Return empty DataFrame with expected structure if no data
        return pd.DataFrame()


df = pd.read_csv("data/liked.csv")
spotify_ids = df["Track URI"].str.split(":").str[-1].to_list()
df, stats = get_track_data_dataframe(spotify_ids)

df.to_csv("data/recco_beat.csv")

features = get_track_audio_features(
    df["recco_beat_id"].to_list(),
    delay_between_requests=0.3,  # 300ms between requests
    max_retries=5,
    verbose=True,
)

features.to_csv("data/recco_beat_audio_features.csv")

features = features.merge(df, left_on="id", right_on="api_id", how="left")

features.to_csv("data/complete_features.csv")

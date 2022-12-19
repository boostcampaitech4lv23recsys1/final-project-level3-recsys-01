import requests
from bs4 import BeautifulSoup
import argparse
import pandas as pd
import time
from tqdm import tqdm
import os


def save_csv(user_infos, check_start, check_end, save_path):
    print(f"---------------save {check_start} {check_end}---------------")
    print(f"---------------user_infos len {len(user_infos)}---------------")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    df = pd.DataFrame(user_infos, columns=["server", "user_info_url", "user"])
    df.to_csv(os.path.join(save_path, f"user_info_{check_start}_{check_end}.csv"), index=False)
    return list()


def crawler(delay, save_every, start_iter, end_iter, save_path):
    user_infos = list()
    check_start = start_iter
    for i in tqdm(range(start_iter, end_iter + 1)):
        req = requests.get(f"https://maple.gg/rank/total?page={i}")
        soup = BeautifulSoup(req.text, "html.parser")
        users = soup.findAll("div", "d-inline-block align-middle")

        for user in users:
            user_info = list()
            user_info.append(user.select("img")[0].get("alt"))
            user_info.append(user.select("a")[0].get("href"))
            user_info.append(user.select("a")[0].text)
            user_infos.append(user_info)

        if (i * 20) % save_every == 0:
            user_infos = save_csv(user_infos, check_start, (i * 20), save_path)
            check_start = (i * 20)

        if delay:
            time.sleep(delay)

    save_csv(user_infos, check_start, (i * 20), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--delay", default=0, type=int)
    parser.add_argument("-s", "--save_every", default=10000, type=int)
    parser.add_argument("--start_iter", default=1, type=int)
    parser.add_argument("--end_iter", default=10000, type=int)
    parser.add_argument("--save_path", default="./data", type=str)
    args = parser.parse_args()

    crawler(args.delay, args.save_every, args.start_iter, args.end_iter, args.save_path)

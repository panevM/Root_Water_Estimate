"""Самостојна еднодневна проверка на API преку acquisition.acquire.

CLI бара --date во YYYY-MM-DD и нова --output патека; --env-file по избор
вчитува пристапни податоци. Ги заменува само датумите од load_config,
прави вистински мрежни барања и запишува кеш преку acquire. Печати статус,
мрежа, прифатени NDMI денови и грешки; излезниот код е 0 за complete,
инаку 2. Празен ден не се пополнува со измислени набљудувања.
"""
import argparse
from root_zone_water.acquisition import acquire
from root_zone_water.config import load_config

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--date',required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
    c=load_config()|{'start_date':args.date,'end_date':args.date}
    m=acquire(c,args.output)
    print('Status:',m['status'])
    print('Requested grid:',m['grid'])
    print('Usable NDMI days:',m.get('usable_ndmi_days',0))
    print('Errors:',m['errors'])
    raise SystemExit(0 if m['status']=='complete' else 2)

"""Small real API response check. No fabricated observations on an empty day."""
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

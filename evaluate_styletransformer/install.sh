wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1szwxIW4wBFH51njpYn6x30pOgg_pOj-G' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1szwxIW4wBFH51njpYn6x30pOgg_pOj-G" -O ppl.tar.gz && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1C2b3B-XzO7kkryUmbwmM9ute0RJoiTxw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1C2b3B-XzO7kkryUmbwmM9ute0RJoiTxw" -O style.tar.gz && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sBWb5UTYUqeHVgiQvTRhJg0QTCdRGTI0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sBWb5UTYUqeHVgiQvTRhJg0QTCdRGTI0" -O style.tar.gz && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12be-obegNfaLen_iaXLs9Jwol7MB2pkJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12be-obegNfaLen_iaXLs9Jwol7MB2pkJ" -O ppl_yelp.binary.tar.gz && rm -rf /tmp/cookies.txt

tar -zxvf ppl.tar.gz
tar -zxvf style.tar.gz
tar -zxvf ppl_yelp.binary.tar.gz 
mv ppl_yelp.binary evaluator
# pip install -r requirements.txt
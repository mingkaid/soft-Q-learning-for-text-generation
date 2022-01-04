wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=11rci8TdDJ8qG4KsH1-6eUV4omhMEfwFz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11rci8TdDJ8qG4KsH1-6eUV4omhMEfwFz" -O dataset.tar.gz && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1szwxIW4wBFH51njpYn6x30pOgg_pOj-G' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1szwxIW4wBFH51njpYn6x30pOgg_pOj-G" -O ppl.tar.gz && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1C2b3B-XzO7kkryUmbwmM9ute0RJoiTxw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1C2b3B-XzO7kkryUmbwmM9ute0RJoiTxw" -O style.tar.gz && rm -rf /tmp/cookies.txt

tar -zxvf dataset.tar.gz
tar -zxvf ppl.tar.gz
tar -zxvf style.tar.gz
pip install -r requirements.txt
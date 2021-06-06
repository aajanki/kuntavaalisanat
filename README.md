# Puolueille ominaiset sanat kuntavaaleissa 2021

Tämä analyysi pyrkii vastaamaan kysymykseen, mitkä sanat esiintyvät usein vain yhden puolueen ehdokkaiden vaalilupauksissa.

[Tulokset](results/kuntavaalilupaukset2021.pdf)

## Analyysin suorittaminen

```
mkdir -p data/vaalikone
wget --directory-prefix data/vaalikone https://github.com/raspi/scrapy-yle-kuntavaalit2021/releases/download/v1.0.0/items.zip

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m kuntavaalit.analyysi
```

## License

[The MIT license](LICENSE)

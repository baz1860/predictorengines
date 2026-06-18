#!/usr/bin/env python3
"""Replace the 12 ea_proxy squads in data/squads.csv with official 2026 lists.

Source: Wikipedia "2026 FIFA World Cup squads" (FIFA tournament squad
announcements), captured via browser on 2026-06-14. Each team's 26-man list with
GK/DF/MF/FW positions. Run once; re-run squads.py afterwards.
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
SQUADS = HERE / "data" / "squads.csv"

# team -> "POS Name;POS Name;..." (26 each), official 2026 squads
DATA = {
"Iran": "GK Alireza Beiranvand;DF Saleh Hardani;DF Ehsan Hajsafi;DF Shojae Khalilzadeh;DF Milad Mohammadi;MF Saeid Ezatolahi;MF Alireza Jahanbakhsh;MF Mohammad Mohebi;FW Mehdi Taremi;FW Mehdi Ghayedi;FW Ali Alipour;GK Payam Niazmand;DF Hossein Kanaanizadegan;MF Saman Ghoddos;MF Rouzbeh Cheshmi;MF Mehdi Torabi;DF Aria Yousefi;FW Amirhossein Hosseinzadeh;DF Ali Nemati;FW Shahriyar Moghanlou;MF Mohammad Ghorbani;GK Hossein Hosseini;DF Ramin Rezaeian;FW Dennis Eckert;DF Danial Eiri;MF Amirmohammad Razzaghinia",
"Iraq": "GK Fahad Talib;DF Rebin Sulaka;DF Hussein Ali;DF Zaid Tahseen;DF Akam Hashim;DF Manaf Younis;MF Youssef Amyn;MF Ibrahim Bayesh;FW Ali Al-Hamadi;FW Mohanad Ali;FW Ahmed Qasem;GK Jalal Hassan;FW Ali Yousif;MF Zidane Iqbal;DF Ahmed Maknzi;MF Amir Al-Ammari;FW Ali Jasim;FW Aymen Hussein;MF Kevin Yakob;MF Aimar Sher;FW Marko Farji;GK Ahmed Basil;DF Merchas Doski;MF Zaid Ismail;DF Mustafa Saadoon;DF Frans Putros",
"Jordan": "GK Yazeed Abulaila;DF Mohammad Abu Hashish;DF Abdallah Nasib;DF Husam Abu Dahab;DF Yazan Al-Arab;MF Amer Jamous;FW Mohammad Abu Zrayq;MF Noor Al-Rawabdeh;FW Ali Olwan;FW Musa Al-Taamari;FW Odeh Al-Fakhouri;GK Nour Bani Attiah;FW Mahmoud Al-Mardi;MF Rajaei Ayed;MF Ibrahim Sadeh;DF Mo Abualnadi;DF Salim Obaid;MF Mohammad Taha;DF Saed Al-Rosan;MF Mohannad Abu Taha;MF Nizar Al-Rashdan;GK Abdallah Al-Fakhouri;DF Ihsan Haddad;FW Ali Azaizeh;MF Mohammad Al-Dawoud;DF Anas Badawi",
"New Zealand": "GK Max Crocombe;DF Tim Payne;DF Francis de Vries;DF Tyler Bindon;DF Michael Boxall;MF Joe Bell;MF Matthew Garbett;MF Marko Stamenić;FW Chris Wood;MF Sarpreet Singh;MF Elijah Just;GK Alex Paulsen;DF Liberato Cacace;MF Alex Rufer;DF Nando Pijnaker;DF Finn Surman;FW Kosta Barbarouses;FW Ben Waine;MF Ben Old;MF Callum McCowatt;FW Jesse Randall;GK Michael Woud;MF Ryan Thomas;DF Callan Elliot;MF Lachlan Bayliss;DF Tommy Smith",
"Norway": "GK Ørjan Nyland;MF Morten Thorsby;DF Kristoffer Ajer;DF Leo Østigård;DF David Møller Wolfe;MF Patrick Berg;FW Alexander Sørloth;MF Sander Berge;FW Erling Haaland;MF Martin Ødegaard;FW Jørgen Strand Larsen;GK Sander Tangvik;GK Egil Selvik;MF Fredrik Aursnes;DF Fredrik André Bjørkan;DF Marcus Holmgren Pedersen;DF Torbjørn Heggem;MF Kristian Thorstvedt;MF Thelo Aasgaard;MF Antonio Nusa;MF Andreas Schjelderup;MF Oscar Bobb;MF Jens Petter Hauge;DF Sondre Langås;DF Henrik Falchener;DF Julian Ryerson",
"Panama": "GK Luis Mejía;DF César Blackman;DF José Córdoba;DF Fidel Escobar;DF Edgardo Fariña;MF Cristian Martínez;MF José Luis Rodríguez;MF Adalberto Carrasquilla;FW Tomás Rodríguez;MF Ismael Díaz;MF Yoel Bárcenas;GK César Samudio;DF Jiovany Ramos;DF Carlos Harvey;DF Eric Davis;DF Andrés Andrade;FW José Fajardo;FW Cecilio Waterman;MF Alberto Quintero;MF Aníbal Godoy;MF César Yanis;GK Orlando Mosquera;DF Michael Amir Murillo;FW Azarias Londoño;DF Roderick Miller;DF Jorge Gutiérrez",
"Portugal": "GK Diogo Costa;DF Nélson Semedo;DF Rúben Dias;DF Tomás Araújo;DF Diogo Dalot;MF Matheus Nunes;FW Cristiano Ronaldo;MF Bruno Fernandes;FW Gonçalo Ramos;MF Bernardo Silva;FW João Félix;GK José Sá;DF Renato Veiga;DF Gonçalo Inácio;MF João Neves;FW Francisco Trincão;FW Rafael Leão;FW Pedro Neto;FW Gonçalo Guedes;DF João Cancelo;MF Rúben Neves;GK Rui Silva;MF Vitinha;DF Samú Costa;DF Nuno Mendes;FW Francisco Conceição",
"Saudi Arabia": "GK Nawaf Al-Aqidi;DF Ali Majrashi;DF Ali Lajami;DF Abdulelah Al-Amri;DF Hassan Al-Tambakti;MF Nasser Al-Dawsari;MF Musab Al-Juwayr;FW Ayman Yahya;FW Firas Al-Buraikan;FW Salem Al-Dawsari;FW Saleh Al-Shehri;DF Saud Abdulhamid;DF Nawaf Boushal;DF Hassan Kadesh;MF Abdullah Al-Khaibari;MF Ziyad Al-Johani;FW Khalid Al-Ghannam;MF Alaa Al-Hejji;FW Abdullah Al-Hamdan;FW Sultan Mandash;GK Mohammed Al-Owais;GK Ahmed Al-Kassar;MF Mohamed Kanno;DF Moteb Al-Harbi;DF Jehad Thakri;DF Mohammed Abu Al-Shamat",
"Senegal": "GK Yehvann Diouf;DF Mamadou Sarr;DF Kalidou Koulibaly;DF Abdoulaye Seck;MF Idrissa Gueye;MF Pathé Ciss;FW Assane Diao;MF Lamine Camara;FW Bamba Dieng;FW Sadio Mané;FW Nicolas Jackson;FW Cherif Ndiaye;FW Iliman Ndiaye;DF Ismail Jakobs;DF Krépin Diatta;GK Édouard Mendy;MF Pape Matar Sarr;FW Ismaïla Sarr;DF Moussa Niakhaté;FW Ibrahim Mbaye;MF Habib Diarra;MF Bara Sapoko Ndiaye;GK Mory Diaw;DF Antoine Mendy;DF El Hadji Malick Diouf;MF Pape Gueye",
"Spain": "GK David Raya;DF Marc Pubill;DF Álex Grimaldo;DF Eric García;DF Marcos Llorente;MF Mikel Merino;FW Ferran Torres;MF Fabián Ruiz;MF Gavi;FW Dani Olmo;FW Yéremy Pino;DF Pedro Porro;GK Joan Garcia;DF Aymeric Laporte;MF Álex Baena;MF Rodri;FW Nico Williams;MF Martín Zubimendi;FW Lamine Yamal;MF Pedri;FW Mikel Oyarzabal;DF Pau Cubarsí;GK Unai Simón;DF Marc Cucurella;FW Víctor Muñoz;FW Borja Iglesias",
"Uruguay": "GK Sergio Rochet;DF José María Giménez;DF Sebastián Cáceres;DF Ronald Araújo;MF Manuel Ugarte;MF Rodrigo Bentancur;MF Nicolás de la Cruz;MF Federico Valverde;FW Darwin Núñez;MF Giorgian de Arrascaeta;FW Facundo Pellistri;GK Santiago Mele;DF Guillermo Varela;MF Agustín Canobbio;MF Emiliano Martínez;DF Mathías Olivera;DF Matías Viña;FW Brian Rodríguez;FW Rodrigo Aguirre;MF Maximiliano Araújo;FW Federico Viñas;MF Joaquín Piquerez;GK Fernando Muslera;DF Santiago Bueno;MF Juan Manuel Sanabria;MF Rodrigo Zalazar",
"Uzbekistan": "GK Utkir Yusupov;DF Abdukodir Khusanov;DF Khojiakbar Alijonov;DF Farrukh Sayfiev;DF Rustam Ashurmatov;MF Akmal Mozgovoy;MF Otabek Shukurov;MF Jamshid Iskanderov;MF Odiljon Hamrobekov;MF Jaloliddin Masharipov;MF Oston Urunov;GK Abduvohid Nematov;DF Sherzod Nasrullaev;FW Eldor Shomurodov;DF Umar Eshmurodov;GK Botirali Ergashev;MF Dostonbek Khamdamov;DF Abdulla Abdullaev;MF Azizjon Ganiev;FW Azizbek Amonov;FW Igor Sergeev;MF Abbosbek Fayzullaev;MF Sherzod Esanov;DF Bekhruz Karimov;DF Avazbek Ulmasaliev;DF Jakhongir Urozov",
}


def main():
    df = pd.read_csv(SQUADS)
    proxy_teams = set(df[df["source"] == "ea_proxy"]["team"])
    assert proxy_teams == set(DATA), f"mismatch: {proxy_teams ^ set(DATA)}"

    kept = df[df["source"] != "ea_proxy"].copy()
    new_rows = []
    for team, blob in DATA.items():
        for entry in blob.split(";"):
            pos, name = entry.split(" ", 1)
            assert pos in ("GK", "DF", "MF", "FW"), entry
            new_rows.append({"team": team, "pos": pos, "player": name.strip(),
                             "source": "wiki"})
    out = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.sort_values(["team", "source"]).reset_index(drop=True)
    out.to_csv(SQUADS, index=False)
    print(f"squads.csv: removed {len(df) - len(kept)} ea_proxy rows, added "
          f"{len(new_rows)} official rows -> {len(out)} total, "
          f"{out['team'].nunique()} teams")
    print("source counts:", out["source"].value_counts().to_dict())
    print("ea_proxy remaining:", (out["source"] == "ea_proxy").sum())


if __name__ == "__main__":
    main()

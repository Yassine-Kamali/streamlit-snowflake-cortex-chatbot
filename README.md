# Chatbot Streamlit + Snowflake Cortex

Application web conversationnelle de type ChatGPT, hebergee dans **Streamlit in Snowflake**, utilisant **Snowflake Cortex** sans cle OpenAI.

## Architecture
- Frontend: Streamlit (`app.py`) execute dans Snowflake.
- Orchestration: Snowpark session active (`get_active_session`).
- LLM: appel SQL `SNOWFLAKE.CORTEX.TRY_COMPLETE` (gestion d'erreurs plus robuste).
- Persistance: table Snowflake `CHAT_MESSAGES`.

Flux:
1. L'utilisateur envoie un message via `st.chat_input`.
2. L'app construit le prompt avec message `system` + historique.
3. L'app appelle Cortex avec le modele selectionne et la temperature.
4. La reponse est affichee puis stockee en base.

## Arborescence
```text
.
|-- app.py
|-- README.md
`-- sql
    |-- 01_setup_environment.sql
    `-- 02_create_conversation_table.sql
```

## Partie A - Mise en place Snowflake
Executer:
1. `sql/01_setup_environment.sql`
2. `sql/02_create_conversation_table.sql`
3. Dans `sql/01_setup_environment.sql`, remplacer `<YOUR_ROLE>` par votre role Snowflake existant.

Resultat attendu:
- Warehouse: `WH_LAB`
- Database: `DB_LAB`
- Schema: `CHAT_APP`
- Role: un role existant (`<YOUR_ROLE>`)
- Table: `CHAT_MESSAGES`
- Cross-region Cortex active: `CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION'`

## Partie B - Interface Chat
L'interface implemente:
- Titre + description.
- Zone d'affichage des messages.
- Saisie utilisateur avec `st.chat_input`.
- Sidebar avec:
  - selection du modele Cortex,
  - slider `temperature` (0.0 -> 1.5),
  - bouton `Nouveau Chat`.
- Etat via `st.session_state` avec format:
  - `{"role": "user/assistant/system", "content": "..."}`
- Le role `system` est stocke mais non affiche dans la conversation.

## Partie C - Integration Cortex
- Prompt construit a partir de:
  - instruction systeme,
  - historique tronque de la conversation.
- Appel LLM via Snowflake uniquement:

```sql
SELECT SNOWFLAKE.CORTEX.TRY_COMPLETE(
  :model,
  PARSE_JSON(:history_json),
  PARSE_JSON(:options_json)
)
```

- Parametres transmis:
  - modele selectionne,
  - temperature,
  - historique complet (apres troncature).

## Partie D - Persistance
Schema conforme au sujet:
- `conversation_id STRING`
- `user_name STRING` (ajoute pour isoler les conversations par utilisateur)
- `timestamp TIMESTAMP`
- `role STRING`
- `content STRING`

Fonctionnalites implementees:
- generation d'un `conversation_id` (UUID) par chat,
- detection de `CURRENT_USER()` et filtrage des conversations par utilisateur,
- insertion des messages `system`, `user`, `assistant`,
- rechargement automatique d'une conversation depuis la liste `Mes conversations` (sans bouton `Charger`).
- nom des conversations base sur le premier message utilisateur.
- export de la conversation courante en JSON et CSV.

## Deploiement Streamlit in Snowflake
1. Ouvrir Snowflake > **Projects** > **Streamlit** > **Create app**.
2. Choisir `DB_LAB.CHAT_APP` et `WH_LAB`.
3. Coller le contenu de `app.py` dans l'editeur Streamlit Snowflake.
4. Lancer l'application.

## Depannage region modele
Si vous voyez l'erreur:
- `The model you requested is unavailable in your region`

Alors executez (avec `ACCOUNTADMIN`):
```sql
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';
```
Valeurs possibles: `AWS_US`, `AWS_EU`, `AWS_APJ`, `ANY_REGION`.

## Choix techniques
- Modele par defaut: `claude-3-5-sonnet`.
- Liste des modeles:
  - tentative de lecture dynamique depuis `SNOWFLAKE.MODELS`,
  - fallback sur une liste de modeles Cortex connus.
- Gestion taille historique:
  - slider `Tours d'historique conserves`,
  - conservation des `N` derniers tours (user+assistant),
  - message `system` toujours conserve.

## Reponses aux questions de validation
1. **Quel modele Cortex avez-vous utilise et pourquoi ?**
J'ai utilise `claude-3-5-sonnet` par defaut pour un bon compromis qualite/cout/latence, avec possibilite de changer le modele dans la sidebar.

2. **Comment gerez-vous la taille de l'historique ?**
L'app tronque l'historique au nombre de tours choisi dans la sidebar. Cela limite les tokens et stabilise les temps de reponse.

3. **Comment avez-vous construit le prompt ?**
Le prompt est une liste de messages structures (`system`, puis `user/assistant`) envoyee en JSON a `SNOWFLAKE.CORTEX.TRY_COMPLETE`.

4. **Quelles difficultes techniques avez-vous rencontrees ?**
Les principales difficultes sont les privileges Cortex/Streamlit et la robustesse de parsing des reponses selon le format retourne par le modele.

5. **Comment garantir la confidentialite des conversations stockees ?**
- Limiter les roles ayant `SELECT` sur la table.
- Isoler les donnees par schema/role.
- Eviter de stocker des donnees sensibles non necessaires.
- Appliquer les politiques de securite Snowflake (RBAC, audit, masquage si besoin).

## Livrables a fournir
- Lien de l'application deployee (a renseigner).
- Capture d'ecran de l'app fonctionnelle (a ajouter dans le repo).
- Repository GitHub public contenant ce code et cette documentation.
- Envoi du lien GitHub a `axel@logbrain.fr`.

## Commandes utiles de verification
```bash
python -m py_compile app.py
```

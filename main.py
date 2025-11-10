import sqlite3
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Optional, List
import asyncio

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import pipeline
from tavily import TavilyClient
from google.genai import Client
from langdetect import detect
from googletrans import Translator
from dotenv import load_dotenv

# ==================== CONFIGURATION ====================

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GENAI_KEY = os.getenv("GEMINI_API_KEY", "")

# Chargement configuration
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

OFFICIAL_SOURCES = config["OFFICIAL_SOURCES"]

# Logging
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Chargement modèles
logger.info("Chargement du classificateur XLM-RoBERTa...")
classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    tokenizer="xlm-roberta-large",
    use_fast=False
)

translator_global = Translator()

logger.info("Tous les modèles chargés avec succès ✅")


# ==================== MODÈLES PYDANTIC ====================

class Question(BaseModel):
    text: str
    session_id: Optional[str] = None



# =============== TRAITEMENT DES LANGUES =================

async def first_lang(text):
    """Traduction asynchrone vers le français"""
    original_lang = detect(text)

    if original_lang != 'fr':
        # Exécuter la traduction dans un thread séparé
        translated_text = await asyncio.to_thread(
            translator_global.translate, text, dest='fr'
        )
        return translated_text.text, original_lang
    else:
        return text, original_lang


async def last_lang(text, original_lang):
    """Traduction asynchrone vers la langue originale"""
    if original_lang != 'fr':
        translated_text = await asyncio.to_thread(
            translator_global.translate, text, dest=original_lang
        )
        return translated_text.text
    return text


# ==================== BASE DE DONNÉES ====================

def init_db():
    """Initialise la base SQLite pour l'historique"""
    connection = sqlite3.connect('memory.db')
    c = connection.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            query TEXT,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            is_admin_topic BOOLEAN
        )
    ''')
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_session 
        ON history(session_id, timestamp DESC)
    ''')
    connection.commit()
    logger.info("Base de donnees initialisee")

init_db()


# ==================== AGENTS ====================

class BinaryClassificationAgent:
    """
    Agent de classification binaire pour filtrer les questions hors-sujet.
    Confiance élevée (0.85-0.98) vs classification multi-classes (0.25-0.40).
    """

    def __init__(self):
        self.classifier = classifier
        self.labels = [
            "question sur démarches administratives ou services publics béninois",
            "question hors sujet administration, conversation générale ou autre"
        ]
        self._cache = {}
        self._cache_ttl = 1800  # 30min

    async def execute(self, text: str) -> Dict:
        """
        Classifie la question en admin/hors-sujet.

        Returns:
            {
                "is_admin_topic": bool,
                "confidence": float,
                "label": str
            }
        """
        cache_key = text.lower().strip()

        # Cache check
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.info(f"Classification cache hit")
                return cached_data

        try:

            result = self.classifier(text, self.labels, multi_label=False)

            is_admin = result['labels'][0] == self.labels[0]
            confidence = result['scores'][0]

            # Boost pour mots-clés administratifs évidents
            admin_keywords = [
                'passeport', 'cni', 'carte', 'identité', 'naissance', 'acte',
                'mariage', 'permis', 'conduire', 'impôts', 'taxe', 'douane',
                'entreprise', 'rccm', 'visa', 'certificat', 'attestation',
                'déclaration', 'dédouanement', 'immatriculation', 'greffier'
            ]

            if any(kw in text.lower() for kw in admin_keywords):
                confidence = min(1.0, confidence + 0.15)
                is_admin = True

            result_data = {
                "is_admin_topic": is_admin,
                "confidence": confidence,
                "label": result['labels'][0]
            }

            # Cache store
            self._cache[cache_key] = (result_data, time.time())

            logger.info(
                f"Classification: {'ADMIN' if is_admin else 'HORS-SUJET'} "
                f"(conf: {confidence:.2f})"
            )

            return result_data

        except Exception as e:
            logger.error(f"Erreur classification: {e}")
            # Fallback conservateur
            return {
                "is_admin_topic": True,  # On laisse passer en cas de doute
                "confidence": 0.5,
                "label": "erreur_classification"
            }


class SmartSearchAgent:
    """
    Agent de recherche intelligent via Tavily.
    Pas d'intent prédéfini : on laisse Tavily comprendre la requête.
    """

    def __init__(self):
        self.client = TavilyClient(api_key=TAVILY_API_KEY)
        self._cache = {}
        self._cache_ttl = 86400 # 24h pour infos stables


    async def execute(
            self,
            user_query: str,
            force_refresh: bool = False
    ) -> Dict:

        search_query = user_query.strip()
        cache_key = search_query.lower()


        # Cache check
        if not force_refresh and cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.info(f"Recherche cache hit: {cache_key[:50]}...")
                return cached_data

        try:
            logger.info(f"Recherche Tavily: {search_query}")

            # Recherche asynchrone
            def _do_search():
                return self.client.search(
                    query=search_query,
                    max_results=5,
                    include_domains=OFFICIAL_SOURCES,
                    search_depth="advanced",
                    include_images=False
                )

            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, _do_search),
                timeout=20
            )

            if not response.get("results"):
                logger.warning("Tavily: Aucun résultat")
                return self._fallback(user_query)

            result = self._process_results(response)

            # Cache store
            self._cache[cache_key] = (result, time.time())

            logger.info(f"Tavily: {len(result['sources'])} sources trouvées")
            return result

        except asyncio.TimeoutError:
            logger.error("Tavily timeout (20s)")
            return self._fallback(user_query)

        except Exception as e:
            logger.error(f"Erreur Tavily: {e}")
            return self._fallback(user_query)

    @staticmethod
    def _process_results(response: Dict) -> Dict:

        """Agrège les résultats Tavily"""
        results = response.get("results", [])

        # Agrège TOUS les contenus (pas juste le premier)
        all_content = "\n\n--- NOUVELLE SOURCE ---\n\n".join([
            f"📄 Source: {r['url']}\n\n{r['content']}"
            for r in results[:5]
            if r.get('content')
        ])

        sources = [r["url"] for r in results if r.get("url")]

        return {
            "content": all_content,
            "sources": sources,
            "raw_results": results
        }

    @staticmethod
    def _fallback(query: str) -> Dict:
        """Fallback si Tavily échoue"""
        return {
            "content": f"Aucune information officielle trouvée pour : '{query}'",
            "sources": [],
            "raw_results": []
        }


class SyntheseAgent:
    """
    Agent de synthèse avec Gemini 2.5 Flash.
    Compatible google-genai v1.46.0.
    Utilise le chat endpoint pour générer du texte et renvoyer un JSON structuré.
    """

    def __init__(self, api_key: str):
        self.client = Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    async def execute(self, search_results: Dict, user_query: str) -> Dict:
        content = search_results.get("content", "")

        if not content or len(content) < 100:
            return self._create_empty_response(search_results)

        prompt = self._build_prompt(user_query, content)

        try:
            logger.info("Appel Gemini ...")
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt
            )

            # Récupération du texte généré
            text_output = response.text
            logger.info(f"Réponse brute Gemini: {text_output}")

            # Parsing JSON
            try:
                result = json.loads(text_output.strip())
            except json.JSONDecodeError:
                logger.warning("Impossible de parser JSON Gemini, fallback appliqué")
                return self._create_fallback_response(text_output, search_results)

            result = self._clean_result(result)




            # Mapping final vers les clés attendues
            mapped_result = {
                "reponse": result.get("reponse", ""),
                "pieces_requises": result.get("pieces_requises"),
                "cout": result.get("cout"),
                "delai_traitement": result.get("delai_traitement"),
                "lieux": result.get("lieux"),
                "etapes": result.get("etapes"),
                "sources": result.get("sources", search_results.get("sources", []))
            }

            logger.info("✅ Synthèse réussie")
            return mapped_result

        except Exception as e:
            logger.error(f"Erreur Gemini: {e}", exc_info=True)
            return self._create_fallback_response(content, search_results)

    @staticmethod
    def _build_prompt(query: str, content: str) -> str:
        return f"""Tu es un assistant spécialisé dans l'extraction d'informations administratives béninoises.

    **QUESTION DE L'UTILISATEUR:**
    {query}

    **CONTENU DES SOURCES OFFICIELLES:**
    {content[:4000]}

    **TA MISSION:**
    1. Lis attentivement le contenu ci-dessus
    2. Extrais TOUTES les informations structurées (pièces, coût, délai, lieux, étapes)
    3. Réponds à la question en 2-4 phrases claires
    4. Retourne un JSON valide sans ```json ```

    **EXEMPLE DE BONNE EXTRACTION:**

    Contenu brut: "Pièces requises: CNI, Acte de naissance. Coût: 25000 FCFA. Délai: 15 jours."

    Réponse JSON attendue:
    {{
      "reponse": "Pour obtenir le document, vous devez fournir votre CNI et votre acte de naissance. Le coût est de 25 000 FCFA et le traitement prend 15 jours.",
      "pieces_requises": ["CNI", "Acte de naissance"],
      "cout": "25 000 FCFA",
      "delai_traitement": "15 jours",
      "lieux": null,
      "etapes": null,
      "sources": ["https://service-public.bj"]
    }}

    **TON TOUR MAINTENANT:**

    Analyse le contenu ci-dessus et retourne UN SEUL JSON avec ces clés EXACTES:
    - "reponse" (string): Réponse synthétique en 2-4 phrases
    - "pieces_requises" (array ou null): Liste de TOUS les documents mentionnés
    - "cout" (string ou null): Montant EXACT en FCFA
    - "delai_traitement" (string ou null): Durée EXACTE
    - "lieux" (array ou null): Liste des adresses/services
    - "etapes" (array ou null): Liste des étapes numérotées
    - "sources" (array): URLs des sources

    **RÈGLES ABSOLUES:**
    ✓ EXTRAIS tout, n'invente rien
    ✓ Utilise null (pas "", pas []) pour infos absentes
    ✓ PAS de copier-coller du texte brut dans "reponse"
    ✓ Commence directement par {{ (pas de texte avant)
    ✓ Termine par }} (pas de texte après)

    JSON:"""
    @staticmethod
    def _clean_result(result: Dict) -> Dict:
        """Nettoie le résultat Gemini"""
        for key in ["pieces_requises", "etapes"]:
            if key in result and isinstance(result[key], list) and not result[key]:
                result[key] = None
        for key in ["cout", "delai_traitement", "lieux"]:
            if key in result and result[key] == "":
                result[key] = None
        return result

    @staticmethod
    def _create_empty_response(search_results: Dict) -> Dict:
        """Réponse quand Tavily n'a rien trouvé"""
        return {
            "reponse": "Désolé, je n'ai trouvé aucune information officielle sur ce sujet dans les sources gouvernementales béninoises.",
            "pieces_requises": None,
            "cout": None,
            "delai_traitement": None,
            "lieux": None,
            "etapes": None,
            "sources": search_results.get("sources", [])
        }

    @staticmethod
    def _create_fallback_response(content: str, search_results: Dict) -> Dict:
        """Fallback si Gemini échoue ou JSON invalide"""
        return {
            "reponse": content[:500] + "..." if len(content) > 500 else content,
            "pieces_requises": search_results.get("pieces_requises"),
            "cout": search_results.get("cout"),
            "delai_traitement": search_results.get("delai_traitement"),
            "lieux": search_results.get("lieux"),
            "etapes": search_results.get("etapes"),
            "sources": search_results.get("sources", [])
        }


class MemoireAgent:
    """Agent de mémorisation des interactions"""

    @staticmethod
    def save_interaction(
            session_id: str,
            query: str,
            response_data: Dict,
            confidence: float,
            is_admin: bool
    ):
        """Sauvegarde l'interaction en base"""
        try:
            with sqlite3.connect('memory.db') as conn:
                c = conn.cursor()
                c.execute(
                    """INSERT INTO history 
                       (session_id, query, response, confidence, is_admin_topic)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        query,
                        json.dumps(response_data, ensure_ascii=False),
                        confidence,
                        is_admin
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Erreur sauvegarde historique: {e}")

    @staticmethod
    def get_history(session_id: str, limit: int = 5) -> List[Dict]:
        """Récupère l'historique d'une session"""
        try:
            with sqlite3.connect('memory.db') as conn:
                c = conn.cursor()
                c.execute(
                    """SELECT query, response, timestamp, confidence 
                       FROM history 
                       WHERE session_id = ? 
                       ORDER BY timestamp DESC 
                       LIMIT ?""",
                    (session_id, limit)
                )
                rows = c.fetchall()
                return [
                    {
                        "query": r[0],
                        "response": json.loads(r[1]),
                        "timestamp": r[2],
                        "confidence": r[3]
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Erreur lecture historique: {e}")
            return []

    @staticmethod
    def cleanup_old_data(days: int = 90):
        """Nettoie les données anciennes"""
        try:
            with sqlite3.connect('memory.db') as conn:
                c = conn.cursor()
                c.execute(
                    f"DELETE FROM history WHERE timestamp < datetime('now', '-{days} days')"
                )
                deleted = c.rowcount
                conn.commit()
                logger.info(f"Nettoyage: {deleted} entrées supprimées (>{days}j)")
        except Exception as e:
            logger.error(f"Erreur nettoyage: {e}")


# ==================== API FASTAPI ====================

app = FastAPI(
    title="Assistant Administratif Bénin (v2 - Architecture Simplifiée)",
    description="Système multi-agents avec classification binaire + Tavily + Gemini",
    version="2.0.0"
)


@app.post("/info")
async def get_info(q: Question):
    """
    Endpoint principal : répond aux questions administratives en multilingue.
    """
    start_time = time.time()

    try:
        # ===== ÉTAPE 0: DÉTECTION LANGUE + TRADUCTION SI NÉCESSAIRE =====
        logger.info(f"🌐 Question reçue: '{q.text}'")

        try:
            question_fr, langue_origine = await first_lang(q.text)
            logger.info(f"🌐 Langue détectée: {langue_origine}")
            logger.info(
                f"🌐 Question traduite vers français: '{question_fr}'" if langue_origine != 'fr' else "🌐 Question déjà en français, pas de traduction nécessaire")
        except Exception as e:
            logger.warning(f"❌ Erreur détection/traduction langue, utilisation texte brut: {e}")
            question_fr = q.text
            langue_origine = 'fr'

        # ===== ÉTAPE 1: CLASSIFICATION BINAIRE =====
        classifier_agent = BinaryClassificationAgent()
        classif = await classifier_agent.execute(question_fr)

        # ===== ÉTAPE 2: FILTRE HORS-SUJET =====
        if not classif["is_admin_topic"] or classif["confidence"] < 0.65:
            logger.info(f"🚫 Question hors-sujet détectée (conf: {classif['confidence']:.2f})")

            # Préparer la réponse hors-sujet (en français d'abord)
            reponse_hors_sujet_fr = (
                "Désolé, je suis spécialisé dans les démarches administratives "
                "et services publics béninois. Je ne peux pas répondre à cette question. "
                "Exemples de ce que je peux vous aider : passeport, CNI, acte de naissance, "
                "permis de conduire, création d'entreprise, impôts, etc."
            )

            # Traduire vers langue originale SI nécessaire
            if langue_origine != 'fr':
                translated_obj = await asyncio.to_thread(
                    translator_global.translate, reponse_hors_sujet_fr, langue_origine
                )
                reponse_finale = translated_obj.text

                logger.info(f"🌐 Réponse hors-sujet traduite vers {langue_origine}")
            else:
                reponse_finale = reponse_hors_sujet_fr

            response_data = {
                "is_admin_topic": False,
                "confidence": classif["confidence"],
                "reponse": reponse_finale,
                "pieces_requises": None,
                "cout": None,
                "delai_traitement": None,
                "lieux": None,
                "etapes": None,
                "sources": [],
                "metadata": {
                    "label": classif["label"],
                    "langue_origine": langue_origine,
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }
            }

            # Sauvegarde historique
            if q.session_id:
                MemoireAgent.save_interaction(
                    q.session_id, q.text, response_data,
                    classif["confidence"], False
                )

            return response_data

        # ===== ÉTAPE 3: RECHERCHE INTELLIGENTE =====
        logger.info("🔍 Lancement de la recherche Tavily...")
        search_agent = SmartSearchAgent()
        search_results = await search_agent.execute(user_query=question_fr)

        # ===== ÉTAPE 4: SYNTHÈSE AVEC GEMINI =====
        logger.info("🧠 Synthèse avec Gemini...")
        synth_agent = SyntheseAgent(api_key=GENAI_KEY)
        synthesis = await synth_agent.execute(search_results, question_fr)

        # ===== ÉTAPE 5: CONSTRUCTION RÉPONSE =====
        execution_time_ms = int((time.time() - start_time) * 1000)

        response_data = {
            "is_admin_topic": True,
            "confidence": classif["confidence"],
            "reponse": synthesis.get("reponse", ""),
            "pieces_requises": synthesis.get("pieces_requises"),
            "cout": synthesis.get("cout"),
            "delai_traitement": synthesis.get("delai_traitement"),
            "lieux": synthesis.get("lieux"),
            "etapes": synthesis.get("etapes"),
            "sources": synthesis.get("sources", []),
            "metadata": {
                "label": classif["label"],
                "langue_origine": langue_origine,
                "execution_time_ms": execution_time_ms,
                "sources_trouvees": len(synthesis.get("sources", []))
            }
        }

        # ===== ÉTAPE 6: TRADUCTION VERS LANGUE ORIGINALE =====
        if langue_origine != 'fr':
            logger.info(f"🌐 Retraduction vers {langue_origine}...")
            try:
                fields_to_translate = []

                if response_data.get("reponse"):
                    fields_to_translate.append("reponse")
                if response_data.get("cout"):
                    fields_to_translate.append("cout")
                if response_data.get("delai_traitement"):
                    fields_to_translate.append("delai_traitement")


                translation_tasks = [
                    last_lang(response_data[field], langue_origine)
                    for field in fields_to_translate
                ]

                # Exécute les traductions en parallèle
                results = await asyncio.gather(*translation_tasks, return_exceptions=True)

                # Met à jour les champs traduits ou log les erreurs
                for field, result in zip(fields_to_translate, results):
                    if isinstance(result, Exception):
                        logger.error(f"Erreur traduction champ '{field}': {result}")
                    else:
                        response_data[field] = result

                logger.info(f"✅ Retraduction réussie vers {langue_origine}")

            except Exception as e:
                logger.error(f"❌ Erreur retraduction: {e}")

        # ===== ÉTAPE 7: MÉMORISATION =====
        if q.session_id:
            MemoireAgent.save_interaction(
                q.session_id, q.text, response_data,
                classif["confidence"], True
            )

        logger.info(
            f"✅ Requête traitée en {execution_time_ms}ms - "
            f"Langue: {langue_origine} - "
            f"Confiance: {classif['confidence']:.2f} - "
            f"Sources: {len(response_data['sources'])}"
        )

        return response_data

    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"❌ Erreur après {execution_time_ms}ms: {e}", exc_info=True)

        # Réponse d'erreur avec traduction si possible
        erreur_message = f"Erreur technique: {str(e)}"
        try:
            # Essayer de détecter la langue depuis la question originale
            detected_lang = detect(q.text)
            if detected_lang != 'fr':
                translated_obj = await asyncio.to_thread(
                    translator_global.translate, erreur_message, detected_lang
                )
                erreur_message = translated_obj.text

        except Exception as trans_error:

            logger.error(f"Erreur détection langue pour erreur: {trans_error}")

        raise HTTPException(
            status_code=500,
            detail=erreur_message
        )

@app.get("/health")
async def health_check():
    """Vérifie l'état du système"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "classifier": "XLM-RoBERTa",
            "search": "Tavily",
            "synthesis": "Gemini 2.5 Flash",
            "database": "SQLite"
        }
    }


@app.get("/sources")
async def get_sources():
    """Liste des sources officielles utilisées"""
    return {
        "official_domains": OFFICIAL_SOURCES,
        "count": len(OFFICIAL_SOURCES)
    }


@app.get("/history/{session_id}")
async def get_session_history(session_id: str, limit: int = 10):
    """Récupère l'historique d'une session"""
    history = MemoireAgent.get_history(session_id, limit)
    return {
        "session_id": session_id,
        "count": len(history),
        "history": history
    }


@app.post("/cleanup")
async def cleanup_database(days: int = 90):
    """Nettoie les données anciennes (admin endpoint)"""
    MemoireAgent.cleanup_old_data(days)
    return {"status": "cleanup_completed", "days": days}


@app.middleware("http")
async def log_performance(request: Request, call_next):
    """Middleware de logging des performances"""
    start_time = time.time()
    response = await call_next(request)
    process_time_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time_ms}ms"
    )

    response.headers["X-Process-Time"] = f"{process_time_ms}ms"
    response.headers["X-Version"] = "2.0.0"

    return response


# ==================== DÉMARRAGE ====================

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Démarrage Assistant Administratif Bénin v2.0")
    logger.info(f"Sources officielles: {len(OFFICIAL_SOURCES)}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
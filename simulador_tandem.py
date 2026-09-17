    """Simulador de rede de filas por eventos discretos - atividade M6.

Gerador congruente linear adotado:
    X(n+1) = (a * X(n) + c) mod M
    U(n+1) = X(n+1) / M
"""

import heapq

A = 1_664_525
C = 1_013_904_223
M = 2**32

_ultimo_x = 42
_aleatorios_restantes = 0
_aleatorios_usados = 0


def InicializarGerador(semente=42, limite=100_000):
    """Reinicia o gerador congruente linear para uma nova simulação."""
    global _ultimo_x, _aleatorios_restantes, _aleatorios_usados

    _ultimo_x = semente
    _aleatorios_restantes = limite
    _aleatorios_usados = 0


def NextRandom():
    """Retorna o próximo pseudoaleatório uniformemente distribuído em [0, 1)."""
    global _ultimo_x, _aleatorios_restantes, _aleatorios_usados

    if _aleatorios_restantes <= 0:
        raise RuntimeError("O limite de números pseudoaleatórios foi atingido.")

    _ultimo_x = (A * _ultimo_x + C) % M
    _aleatorios_restantes -= 1
    _aleatorios_usados += 1

    return _ultimo_x / M


def converter_intervalo(aleatorio, minimo, maximo):
    """Mapeia um valor de [0, 1) para a distribuição uniforme [mínimo, máximo]."""
    return minimo + (maximo - minimo) * aleatorio


def sortear_destino(roteamento):
    """Escolhe o destino de um cliente que termina o atendimento.

    'roteamento' é uma lista de pares (destino, probabilidade), onde o destino
    None representa a saída do cliente da rede. Quando existe um único destino
    com probabilidade 1, nenhum pseudoaleatório é consumido.
    """
    if len(roteamento) == 1 and roteamento[0][1] >= 1.0:
        return roteamento[0][0]

    sorteio = NextRandom()
    acumulado = 0.0

    for destino, probabilidade in roteamento:
        acumulado += probabilidade
        if sorteio < acumulado:
            return destino

    return roteamento[-1][0]


def simular_rede(
    filas,
    chegadas_externas,
    limite_aleatorios=100_000,
    semente=42,
):
    """Executa uma rede de filas G/G/c/K por simulação de eventos discretos.

    'filas' é um dicionário {nome: propriedades}, onde cada fila informa
    servidores, capacidade, atendimento (mínimo e máximo) e roteamento.
    'chegadas_externas' informa, por fila, o intervalo de chegada e o tempo
    da primeira chegada vinda do exterior da rede.
    A capacidade inclui os clientes em atendimento e os que estão esperando.
    Em eventos simultâneos, a saída/passagem tem prioridade sobre a chegada.
    """
    InicializarGerador(semente=semente, limite=limite_aleatorios)

    CHEGADA = "CHEGADA"
    SAIDA = "SAIDA"

    eventos = []
    sequencia = 0

    clientes_no_sistema = {nome: 0 for nome in filas}
    perdas = {nome: 0 for nome in filas}
    tempos_acumulados = {
        nome: [0.0] * (propriedades["capacidade"] + 1)
        for nome, propriedades in filas.items()
    }
    tempo_global = 0.0

    # Prioridade 0 para saída/passagem e 1 para chegada no mesmo instante.
    for nome, chegada in chegadas_externas.items():
        heapq.heappush(
            eventos, (chegada["primeira_chegada"], 1, sequencia, CHEGADA, nome, None)
        )
        sequencia += 1

    def agendar_atendimento(nome, tempo_global):
        """Agenda o término de atendimento de um cliente da fila 'nome'."""
        nonlocal sequencia

        if _aleatorios_restantes <= 0:
            return

        tempo_atendimento = converter_intervalo(
            NextRandom(),
            filas[nome]["atendimento_minimo"],
            filas[nome]["atendimento_maximo"],
        )
        sequencia += 1
        heapq.heappush(
            eventos,
            (tempo_global + tempo_atendimento, 0, sequencia, SAIDA, nome, None),
        )

    while _aleatorios_restantes > 0 and eventos:
        tempo_evento, _, _, tipo, fila, _ = heapq.heappop(eventos)

        # AcumulaTempo: contabiliza o intervalo em TODAS as filas da rede.
        for nome in filas:
            tempos_acumulados[nome][clientes_no_sistema[nome]] += (
                tempo_evento - tempo_global
            )
        tempo_global = tempo_evento

        if tipo == CHEGADA:
            # Agenda a próxima chegada externa desta fila.
            if fila in chegadas_externas and _aleatorios_restantes > 0:
                intervalo_chegada = converter_intervalo(
                    NextRandom(),
                    chegadas_externas[fila]["chegada_minima"],
                    chegadas_externas[fila]["chegada_maxima"],
                )
                sequencia += 1
                heapq.heappush(
                    eventos,
                    (
                        tempo_global + intervalo_chegada,
                        1,
                        sequencia,
                        CHEGADA,
                        fila,
                        None,
                    ),
                )

            if clientes_no_sistema[fila] >= filas[fila]["capacidade"]:
                perdas[fila] += 1
                continue

            clientes_no_sistema[fila] += 1

            if clientes_no_sistema[fila] <= filas[fila]["servidores"]:
                agendar_atendimento(fila, tempo_global)

        else:
            destino = sortear_destino(filas[fila]["roteamento"])

            clientes_no_sistema[fila] -= 1

            # Se ainda há pelo menos 'servidores' clientes no sistema,
            # um cliente que aguardava inicia seu atendimento agora.
            if clientes_no_sistema[fila] >= filas[fila]["servidores"]:
                agendar_atendimento(fila, tempo_global)

            # PASSAGEM: o cliente é tratado como uma chegada na fila destino.
            if destino is not None:
                if clientes_no_sistema[destino] >= filas[destino]["capacidade"]:
                    perdas[destino] += 1
                    continue

                clientes_no_sistema[destino] += 1

                if clientes_no_sistema[destino] <= filas[destino]["servidores"]:
                    agendar_atendimento(destino, tempo_global)

    probabilidades = {
        nome: [
            tempo / tempo_global if tempo_global > 0 else 0.0
            for tempo in tempos_acumulados[nome]
        ]
        for nome in filas
    }

    return {
        "filas": filas,
        "semente": semente,
        "aleatorios_usados": _aleatorios_usados,
        "tempo_global": tempo_global,
        "perdas": perdas,
        "tempos_acumulados": tempos_acumulados,
        "probabilidades": probabilidades,
    }


def formatar_resultado(resultado):
    """Formata os resultados exigidos pela atividade M6."""
    linhas = [
        f"Aleatórios utilizados: {resultado['aleatorios_usados']}",
        "",
    ]

    for nome, propriedades in resultado["filas"].items():
        linhas.append(
            f"{nome} - G/G/{propriedades['servidores']}/{propriedades['capacidade']}"
        )
        linhas.append("Distribuição dos estados:")

        for estado, (tempo, probabilidade) in enumerate(
            zip(resultado["tempos_acumulados"][nome], resultado["probabilidades"][nome])
        ):
            linhas.append(
                f"   Estado {estado}: tempo acumulado = {tempo:.6f}; "
                f"probabilidade = {probabilidade:.8f} "
                f"({probabilidade * 100:.6f}%)"
            )

        linhas.append(f"Clientes perdidos: {resultado['perdas'][nome]}")
        linhas.append("")

    linhas.append(f"Tempo global da simulação: {resultado['tempo_global']:.6f}")

    return "\n".join(linhas)


if __name__ == "__main__":
    filas = {
        "Fila 1": {
            "servidores": 2,
            "capacidade": 3,
            "atendimento_minimo": 4.0,
            "atendimento_maximo": 5.0,
            "roteamento": [("Fila 2", 1.0)],
        },
        "Fila 2": {
            "servidores": 1,
            "capacidade": 5,
            "atendimento_minimo": 1.0,
            "atendimento_maximo": 3.0,
            "roteamento": [(None, 1.0)],
        },
    }

    chegadas_externas = {
        "Fila 1": {
            "chegada_minima": 1.0,
            "chegada_maxima": 5.0,
            "primeira_chegada": 2.5,
        }
    }

    resultado = simular_rede(filas, chegadas_externas)
    print(formatar_resultado(resultado))

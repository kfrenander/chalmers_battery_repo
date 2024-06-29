#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <errno.h>

#define BUFFER_SIZE 1024

void log_message(const char *filename, const char *message){
    FILE *logfile = fopen(filename, "a"); //will create the file if it does not exist
    if (logfile == NULL){
        printf("Failed to open log file");
        exit(EXIT_FAILURE);
    }
    fprintf(logfile, "%s\n", message);
    printf("Temp: %s\n", message);
    fflush(logfile);
    fclose(logfile);
}

int main(int argc, char *argv[]){
    printf("working?\n");
    fflush(stdout);
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <IP> <PORT> <LOGFILE>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    printf("working2\n");
    const char *ip = argv[1];
    int port = atoi(argv[2]);
    const char *logfile = argv[3];

    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[BUFFER_SIZE] = {0};

    // Create socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0){
        printf("Socket creation error");
        exit(EXIT_FAILURE);
    }

    // Set up the server address structure
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    // Convert IP address from text to binary form
    if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) {
        printf("Invalid address/Address is not supported\n");
        exit(EXIT_FAILURE);
    }

    // Connect to the server
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("Connection failed\n");
        exit(EXIT_FAILURE);
    }

    printf("Connected to the server");

    // Receive data from the server
    int read_size;
    while ((read_size = read(sock, buffer, BUFFER_SIZE - 1)) > 0) {
        buffer[read_size] = '\0';
        log_message(logfile, buffer);
    }

    if (read_size == 0) {
        printf("Server closed the connection");
    } else if (read_size < 0) {
        printf("Read error");
    }

    close(sock);
    return 0;
}

